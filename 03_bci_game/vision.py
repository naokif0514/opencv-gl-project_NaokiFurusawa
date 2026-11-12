"""
vision.py
---------
1 台の Web カメラから、ゲームに必要な情報をまとめて取り出す処理スレッド（OpenCV + MediaPipe）。

  ・AR の背景：左右反転したカメラ映像（RGBA）
  ・手（MediaPipe Hands）：位置・速度（02 の hand_tracker.py と同じ計算）と、手の開き具合（把持に使う）
  ・視線の特徴量（MediaPipe Face Mesh、虹彩つき）：gaze.py で画面上の注視点に変換する
  ・色の物（OpenCV の HSV しきい値）：赤・青などの物の位置と大きさ（items.py でアイテムにする）

■ 軽くするための工夫（ソフトウェア描画の PC で、カメラ処理 20 fps 以上を目標にする）
  ・Face Mesh は FACE_EVERY_N フレームに 1 回、Hands は HANDS_EVERY_N フレームに 1 回だけ処理する
    （手が見えていないときの Hands は手のひらの検出が毎回走って重い（約 25 ms）ので、手を優先して毎フレーム、
      視線はもともと平滑化するので 2 フレームに 1 回、を既定にした）
  ・色の検出は、縮小した画像（COLOR_DETECT_WIDTH）で行う（約 0.5 ms）
  ・処理はこの別スレッドで行い、描画のフレームレートに影響しないようにする
"""

import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

import config
from hand_tracker import HandState, VelocityEstimator   # 02 から流用した手の速度の計算


@dataclass
class ColorBlob:
    color: str                 # config.ITEM_COLORS の名前
    pos: np.ndarray            # 中心（カメラ画像の正規化座標 0〜1、左右反転後。y は下向き）
    radius: float              # おおよその半径（画像の幅に対する割合）
    area: float


@dataclass
class VisionState:
    hand: HandState = field(default_factory=HandState)
    hand_open: float | None = None          # 手の開き具合（0 = 握る、1 = 開く）。手がなければ None
    gaze_feat: np.ndarray | None = None     # 視線の特徴量（gaze.py）。顔がなければ None
    face_detected: bool = False
    blobs: list = field(default_factory=list)
    fps: float = 0.0                        # カメラ処理のフレームレート
    face_fps: float = 0.0
    hand_fps: float = 0.0


# ============================================================
# 視線の特徴量（Face Mesh の虹彩と顔の向き）
# ============================================================
# MediaPipe Face Mesh の点の番号（refine_landmarks=True のとき 468〜477 が虹彩）
EYES = (
    # (目尻, 目頭, 上まぶた, 下まぶた, 虹彩の中心)
    (33, 133, 159, 145, 468),      # 画像の左側に写る目
    (263, 362, 386, 374, 473),     # 画像の右側に写る目
)
NOSE_TIP = 1


def gaze_features(lm, w, h):
    """
    Face Mesh の点から、視線の特徴量 [目の中の虹彩の左右, 上下, 顔の向きの左右, 上下] を求める。
      ・虹彩の位置：目尻と目頭を結ぶ線を x 軸として、目の中心からのずれを目の幅で割ったもの（両目の平均）
      ・顔の向き：鼻先が両目の中点からどれだけずれているかを、両目の間の距離で割ったもの
    Web カメラの解像度では虹彩の動きは数画素しかないので、顔の向きも合わせて使う（gaze.py で線形に画面へ対応づける）
    """
    P = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
    iris, centers = [], []
    for outer, inner, top, bottom, iris_i in EYES:
        o, i = P[outer], P[inner]
        ex = i - o
        width = np.linalg.norm(ex)
        if width < 1e-6:
            return None
        ex /= width
        ey = np.array([-ex[1], ex[0]])           # ex を 90° 回した向き（画像の下向きに近い）
        if ey[1] < 0:
            ey = -ey
        c = (o + i) / 2
        d = P[iris_i] - c
        # 左右の目で ex の向き（目尻 → 目頭）が逆なので、画像の右向きにそろえる
        sx = 1.0 if ex[0] > 0 else -1.0
        iris.append((sx * np.dot(d, ex) / width, np.dot(d, ey) / width))
        centers.append(c)
    iris = np.mean(iris, axis=0)
    mid = (centers[0] + centers[1]) / 2
    iod = np.linalg.norm(centers[1] - centers[0])
    nose = P[NOSE_TIP] - mid
    return np.array([iris[0], iris[1], nose[0] / iod, nose[1] / iod])


def hand_openness(lm):
    """
    手の開き具合：指先（人差し指〜小指）と手首の距離の平均を、手のひらの大きさ（手首 → 中指の付け根）で割る。
    握るとおよそ 1.0 以下、開くと 1.8 以上になるので、それを 0〜1 に直す
    """
    P = np.array([(p.x, p.y) for p in lm])
    palm = np.linalg.norm(P[9] - P[0])
    if palm < 1e-6:
        return None
    tips = np.linalg.norm(P[[8, 12, 16, 20]] - P[0], axis=1).mean() / palm
    lo, hi = config.HAND_OPEN_RANGE
    return float(np.clip((tips - lo) / (hi - lo), 0.0, 1.0))


# ============================================================
# 色の物の検出
# ============================================================
def detect_color_blobs(frame_bgr):
    """縮小した画像を HSV に変換し、config.ITEM_COLORS の色の範囲に入る塊を探す"""
    small_w = config.COLOR_DETECT_WIDTH
    scale = small_w / frame_bgr.shape[1]
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    sh, sw = small.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    blobs = []
    for name, spec in config.ITEM_COLORS.items():
        mask = np.zeros((sh, sw), np.uint8)
        for lo, hi in spec["hsv"]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            area = cv2.contourArea(c) / (sw * sh)
            if area < config.COLOR_MIN_AREA:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            found.append(ColorBlob(name, np.array([cx / sw, cy / sh]), r / sw, area))
        found.sort(key=lambda b: -b.area)
        blobs += found[:config.COLOR_MAX_PER_COLOR]
    return blobs


# ============================================================
# 処理スレッド
# ============================================================
class VisionThread(threading.Thread):
    def __init__(self, camera_index=config.CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None
        self._lock = threading.Lock()
        self._state = VisionState()
        self._frame = None            # AR 用の最新のカメラ画像（RGBA）
        self._frame_id = 0
        self._debug = None
        self._running = False
        self.debug_enabled = False
        self.face_enabled = True      # 視線を使わない場面では止めて軽くできる
        self.velocity = VelocityEstimator()

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        ok, _ = self.cap.read()
        if not ok:
            self.cap.release()
            self.cap = None
            return False
        return True

    def stop(self):
        self._running = False
        if self.is_alive():
            self.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def get_state(self):
        with self._lock:
            return self._state

    def get_frame(self):
        """(フレーム番号, RGBA 画像)。番号が変わっていなければテクスチャを送り直さなくてよい"""
        with self._lock:
            return self._frame_id, self._frame

    def pop_debug_frame(self):
        with self._lock:
            f, self._debug = self._debug, None
            return f

    def run(self):
        self._running = True
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        drawing = mp.solutions.drawing_utils
        n = 0
        times = {"all": [], "face": [], "hand": []}
        last_seen = -1e9
        st = VisionState()

        def rate(key, now):
            q = times[key]
            q.append(now)
            while q and now - q[0] > 1.0:
                q.pop(0)
            return float(len(q))

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)                    # 鏡像（自分から見た向きと画面の向きをそろえる）
            now = time.monotonic()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st = VisionState(hand=st.hand, hand_open=st.hand_open, gaze_feat=st.gaze_feat,
                             face_detected=st.face_detected, face_fps=st.face_fps, hand_fps=st.hand_fps)
            hand_lm = face_lm = None

            # ---- 手（02 と同じ：位置・速度。加えて開き具合）----
            if n % config.HANDS_EVERY_N == 0:
                res = hands.process(rgb)
                pos = None
                if res.multi_hand_landmarks:
                    hand_lm = res.multi_hand_landmarks[0]
                    lm = hand_lm.landmark[config.HAND_LANDMARK]
                    pos = np.array([lm.x, 1.0 - lm.y])
                    st.hand_open = hand_openness(hand_lm.landmark)
                    last_seen = now
                detected = pos is not None
                if detected:
                    vel = self.velocity.update(pos, now)
                elif now - last_seen > config.HAND_LOST_SEC:
                    vel = self.velocity.update(None, now)
                    st.hand_open = None
                else:
                    vel = st.hand.vel
                st.hand_fps = rate("hand", now)
                st.hand = HandState(detected=detected, pos=pos, vel=vel, fps=st.hand_fps)

            # ---- 視線の特徴量 ----
            if self.face_enabled and n % config.FACE_EVERY_N == 0:
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    face_lm = res.multi_face_landmarks[0]
                    st.gaze_feat = gaze_features(face_lm.landmark, w, h)
                    st.face_detected = st.gaze_feat is not None
                else:
                    st.gaze_feat, st.face_detected = None, False
                st.face_fps = rate("face", now)

            # ---- 色の物 ----
            st.blobs = detect_color_blobs(frame)
            st.fps = rate("all", now)
            n += 1

            debug = None
            if self.debug_enabled:
                debug = frame.copy()
                if hand_lm is not None:
                    drawing.draw_landmarks(debug, hand_lm, mp.solutions.hands.HAND_CONNECTIONS)
                if face_lm is not None:
                    for idx in (468, 473, 33, 133, 263, 362, NOSE_TIP):
                        p = face_lm.landmark[idx]
                        cv2.circle(debug, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1)
                for b in st.blobs:
                    c = config.ITEM_COLORS[b.color]["draw_bgr"]
                    cv2.circle(debug, (int(b.pos[0] * w), int(b.pos[1] * h)), int(b.radius * w), c, 2)
                text = (f"camera {st.fps:4.1f} fps  face {st.face_fps:4.1f}  hand {st.hand_fps:4.1f}  "
                        f"open {'-' if st.hand_open is None else f'{st.hand_open:.2f}'}")
                cv2.putText(debug, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(debug, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            with self._lock:
                self._state = st
                self._frame = rgba
                self._frame_id += 1
                if debug is not None:
                    self._debug = debug

        face_mesh.close()
        hands.close()
