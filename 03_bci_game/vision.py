"""
vision.py
---------
1 台の Web カメラから、ゲームに必要な情報をまとめて取り出す処理スレッド（OpenCV + MediaPipe）。

  ・AR の背景：左右反転したカメラ映像（RGBA）
  ・手（MediaPipe Hands）：位置・速度（02 の hand_tracker.py と同じ計算）と、手の開き具合（把持に使う）
  ・視線の特徴量（MediaPipe Face Mesh、虹彩つき）：gaze.py で画面上の注視点に変換する
  ・色の物（OpenCV の HSV しきい値）：赤・青などの物の位置と大きさ（items.py でアイテムにする）
      顔の範囲は検出から外す（唇・肌を赤い物と間違えないように）。
      P キーで、画面中央の枠に見せた物の色をその場で登録できる（register_color。照明やカメラで色が変わるため）

■ 軽くするための工夫（ソフトウェア描画の PC で、カメラ処理 20 fps 以上を目標にする）
  ・Face Mesh は FACE_EVERY_N フレームに 1 回（既定は毎フレーム。視線の安定に必要）
  ・Hands は、手を追跡できている間は HANDS_EVERY_N フレームに 1 回（既定は毎フレーム。追跡だけなら約 10 ms と軽い）、
    手を見失っている間は HANDS_SEARCH_EVERY_N フレームに 1 回だけ手を探す（手のひらの検出は約 25 ms と重いため）
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
    gaze_time: float = 0.0                  # その特徴量を計算したフレームの時刻
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
def detect_color_blobs(frame_bgr, colors=None, exclude=None, masks_out=None):
    """
    縮小した画像を HSV に変換し、colors（既定は config.ITEM_COLORS）の色の範囲に入る塊を探す。
    exclude：検出から外す矩形 (x0, y0, x1, y1)（正規化座標）。顔の範囲を入れる
    masks_out：dict を渡すと、色ごとのマスク画像を入れて返す（デバッグ表示用）
    """
    colors = config.ITEM_COLORS if colors is None else colors
    small_w = config.COLOR_DETECT_WIDTH
    scale = small_w / frame_bgr.shape[1]
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    sh, sw = small.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    blobs = []
    for name, spec in colors.items():
        mask = np.zeros((sh, sw), np.uint8)
        for lo, hi in spec["hsv"]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        if exclude is not None:
            x0, y0, x1, y1 = exclude
            mask[max(0, int(y0 * sh)):int(y1 * sh), max(0, int(x0 * sw)):int(x1 * sw)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)      # 物の表面の光の反射でできる小さな穴をふさぐ
        if masks_out is not None:
            masks_out[name] = mask
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


def sample_color(frame_bgr, box=None):
    """
    画像の中央の枠（正規化座標の矩形 box）の色から、その色の HSV の範囲を作る（色の登録）。
    枠の中で彩度の高い画素だけを使い、色相の中央値 ± COLOR_REGISTER_HUE、彩度・明るさは下側 10% 点から下限を決める
    """
    x0, y0, x1, y1 = box or config.COLOR_REGISTER_BOX
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    vivid = hsv[(hsv[:, 1] >= 60) & (hsv[:, 2] >= 40)]
    if len(vivid) < 0.2 * len(hsv):
        return None                                   # 色のはっきりした物が枠に入っていない
    # 色相は 0 と 180 がつながっているので、赤のようにまたがる場合に備えて、中央値は円の上で求める
    ang = vivid[:, 0] * (2 * np.pi / 180)
    hue = int(round((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * 180 / (2 * np.pi))) % 180
    s_lo = int(max(50, np.percentile(vivid[:, 1], 10) * 0.8))
    v_lo = int(max(35, np.percentile(vivid[:, 2], 10) * 0.7))
    dh = config.COLOR_REGISTER_HUE
    ranges = []
    lo, hi = hue - dh, hue + dh
    if lo < 0:
        ranges += [((0, s_lo, v_lo), (hi, 255, 255)), ((180 + lo, s_lo, v_lo), (180, 255, 255))]
    elif hi > 180:
        ranges += [((lo, s_lo, v_lo), (180, 255, 255)), ((0, s_lo, v_lo), (hi - 180, 255, 255))]
    else:
        ranges += [((lo, s_lo, v_lo), (hi, 255, 255))]
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return {"hsv": ranges, "rgb": tuple(float(c) / 255 for c in bgr[::-1]), "draw_bgr": tuple(int(c) for c in bgr)}


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
        self.colors = dict(config.ITEM_COLORS)   # 検出する色（register_color で追加される。差しかえは丸ごと行う）
        self._register_request = None

    def register_color(self, name):
        """次のフレームで、画面中央の枠の色を name として登録する（結果は last_register に入る）"""
        self.last_register = None
        self._register_request = name

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
        face_box = None
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
            st = VisionState(hand=st.hand, hand_open=st.hand_open, gaze_feat=st.gaze_feat, gaze_time=st.gaze_time,
                             face_detected=st.face_detected, face_fps=st.face_fps, hand_fps=st.hand_fps)
            hand_lm = face_lm = None

            # ---- 手（02 と同じ：位置・速度。加えて開き具合）----
            hands_every = config.HANDS_EVERY_N if st.hand.detected else config.HANDS_SEARCH_EVERY_N
            if n % hands_every == 0:
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
                    st.gaze_time = now
                    st.face_detected = st.gaze_feat is not None
                    xs = [p.x for p in face_lm.landmark]
                    ys = [p.y for p in face_lm.landmark]
                    mx, my = (max(xs) - min(xs)) * 0.15, (max(ys) - min(ys)) * 0.15
                    face_box = (min(xs) - mx, min(ys) - my, max(xs) + mx, max(ys) + my)
                else:
                    st.gaze_feat, st.face_detected = None, False
                    face_box = None
                st.face_fps = rate("face", now)

            # ---- 色の物（顔の範囲は除く）----
            if self._register_request is not None:
                spec = sample_color(frame)
                if spec is not None:
                    colors = dict(self.colors)
                    colors[self._register_request] = spec
                    self.colors = colors
                self.last_register = (self._register_request, spec)
                self._register_request = None
            masks = {} if self.debug_enabled else None
            st.blobs = detect_color_blobs(frame, self.colors, face_box, masks)
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
                    c = self.colors[b.color]["draw_bgr"]
                    cv2.circle(debug, (int(b.pos[0] * w), int(b.pos[1] * h)), int(b.radius * w), c, 2)
                if face_box is not None:
                    cv2.rectangle(debug, (int(face_box[0] * w), int(face_box[1] * h)),
                                  (int(face_box[2] * w), int(face_box[3] * h)), (120, 120, 120), 1)
                x0, y0, x1, y1 = config.COLOR_REGISTER_BOX
                cv2.rectangle(debug, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 255), 1)
                # 色ごとのマスク（白 = その色と判定された部分）を右下に小さく並べる
                for k, (name, m) in enumerate(masks.items()):
                    th = cv2.resize(m, (w // 5, h // 5))
                    y = h - th.shape[0] - 4
                    x = w - (k + 1) * (th.shape[1] + 4)
                    debug[y:y + th.shape[0], x:x + th.shape[1]] = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
                    cv2.putText(debug, name, (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                self.colors[name]["draw_bgr"], 1, cv2.LINE_AA)
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
