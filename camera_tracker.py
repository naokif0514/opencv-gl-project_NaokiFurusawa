"""
camera_tracker.py
-----------------
Web カメラから映像を取り込み（OpenCV）、MediaPipe Face Mesh で顔と目の特徴点を検出して、
集中度の材料となる指標を毎フレーム計算する。

計算する指標
  1. まばたき     : EAR（Eye Aspect Ratio）が一定時間しきい値を下回ったら 1 回と数え、
                    直近 60 秒の回数から「回数/分」を出す
  2. 目の開き具合 : EAR の移動平均（眠気の目安）
  3. 顔の向き     : 顔の 6 点と一般的な顔の 3D モデルから cv2.solvePnP で頭部姿勢を推定し、
                    yaw（左右）/ pitch（上下）と正面からのズレ角を出す
  4. 視線の安定性 : 目頭・目尻を基準にした虹彩中心の相対位置の、直近数秒の標準偏差
  5. 離席         : 顔が一定時間検出されない

描画（OpenGL）と別スレッドで動かすことで、カメラの処理速度に描画のフレームレートが
引きずられないようにしている。結果はロック付きで共有し、描画側は最新値を読むだけ。
"""

import math
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import config
from focus_estimator import FocusEstimator, FocusState, Metrics

# ------------------------------------------------------------
# MediaPipe Face Mesh の特徴点番号
# ------------------------------------------------------------
# EAR 用の 6 点 [p1(目尻/目頭), p2, p3(上まぶた), p4(目頭/目尻), p5, p6(下まぶた)]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # 画像の左側に写る目（本人の右目）
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]   # 画像の右側に写る目（本人の左目）

# 視線用：目の両端（画像の左→右の順）と虹彩中心（refine_landmarks=True で取得できる）
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# 頭部姿勢推定用の 6 点：鼻先, あご, 画像左の目尻, 画像右の目尻, 画像左の口角, 画像右の口角
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 上と対応する一般的な顔の 3D モデル座標（単位は任意）。
# カメラ座標系（x: 右, y: 下, z: 奥）に合わせ、正面を向いたとき回転が単位行列になるようにしている。
FACE_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],          # 鼻先
    [0.0, 330.0, 65.0],       # あご
    [-225.0, -170.0, 135.0],  # 画像左の目尻
    [225.0, -170.0, 135.0],   # 画像右の目尻
    [-150.0, 150.0, 125.0],   # 画像左の口角
    [150.0, 150.0, 125.0],    # 画像右の口角
], dtype=np.float64)


def eye_aspect_ratio(pts):
    """
    EAR（Eye Aspect Ratio）を計算する。
        EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)
    目を開いているときは 0.25〜0.35 程度、閉じると 0.1 前後まで下がる。
    """
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal + 1e-6)


def iris_relative_position(corner_a, corner_b, iris):
    """
    目頭・目尻を結ぶ線を基準に、虹彩中心の相対位置 (h, v) を返す。
    h: 目の幅方向の位置（0=左端, 1=右端）、v: それに垂直な方向のずれ
    どちらも目の幅で割っているので、顔とカメラの距離や頭の動きの影響を受けにくい。
    """
    axis = corner_b - corner_a
    width = np.linalg.norm(axis) + 1e-6
    u = axis / width
    d = iris - corner_a
    h = float(np.dot(d, u)) / width
    v = float(u[0] * d[1] - u[1] * d[0]) / width
    return h, v


def rotation_to_yaw_pitch(rvec):
    """solvePnP の回転ベクトルから yaw（左右）と pitch（上下）を度で求める"""
    R, _ = cv2.Rodrigues(rvec)
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
    return yaw, pitch


class CameraTracker(threading.Thread):
    """
    カメラ処理スレッド。
    run() の中で「取り込み → 特徴点検出 → 指標計算 → 集中度更新」を繰り返す。
    """

    def __init__(self, camera_index=config.CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None
        self.estimator = FocusEstimator()

        # 共有データ（描画スレッドから読まれる）
        self._lock = threading.Lock()
        self._metrics = Metrics()
        self._focus = FocusState()
        self._debug_frame = None
        self._running = False

        # 描画側から切り替えるフラグ
        self.debug_enabled = True         # デバッグ画像を作るか
        self._calibrate_request = False   # 「今の顔の向きを正面とする」要求

        # 指標計算用の履歴
        self._start_time = None
        self._last_face_time = None
        self._ear_hist = deque()          # (時刻, EAR)
        self._gaze_hist = deque()         # (時刻, h, v)
        self._blink_times = deque()       # まばたきした時刻
        self._blink_count = 0
        self._eye_closed_since = None     # 目を閉じ始めた時刻（開いているときは None）
        self._yaw0 = 0.0                  # 正面とみなす yaw（キャリブレーションで変更）
        self._pitch0 = 0.0                # 正面とみなす pitch

    # --------------------------------------------------------
    # 起動・停止
    # --------------------------------------------------------
    def open(self):
        """カメラを開く。開けなければ False を返す（テストモードで起動するため）"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        ok, _ = self.cap.read()     # 実際に 1 枚読めるか確認する
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

    def request_calibration(self):
        """現在の顔の向きを「正面」として登録するよう要求する"""
        self._calibrate_request = True

    # --------------------------------------------------------
    # 描画スレッドから最新値を読むための関数
    # --------------------------------------------------------
    def get_state(self):
        """最新の (指標, 集中度) を返す"""
        with self._lock:
            return self._metrics, self._focus

    def pop_debug_frame(self):
        """新しいデバッグ画像があれば返す（一度返したら None に戻す）"""
        with self._lock:
            frame = self._debug_frame
            self._debug_frame = None
            return frame

    # --------------------------------------------------------
    # スレッド本体
    # --------------------------------------------------------
    def run(self):
        self._running = True
        self._start_time = time.monotonic()
        prev = self._start_time
        fps = 0.0

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,          # 虹彩の特徴点（468〜477）も取得する
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            # MediaPipe は RGB 画像を受け取る
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            landmarks = None
            if result.multi_face_landmarks:
                h, w = frame.shape[:2]
                lm = result.multi_face_landmarks[0].landmark
                # 正規化座標(0〜1) → 画素座標
                landmarks = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)

            metrics, pose = self._compute_metrics(landmarks, frame.shape, now)
            metrics.camera_fps = fps
            focus = self.estimator.update(metrics, now)

            debug = None
            if self.debug_enabled:
                debug = self._draw_debug(frame, landmarks, metrics, focus, pose)

            with self._lock:
                self._metrics = metrics
                self._focus = focus
                if debug is not None:
                    self._debug_frame = debug

        face_mesh.close()

    # --------------------------------------------------------
    # 指標の計算
    # --------------------------------------------------------
    def _compute_metrics(self, lm, shape, now):
        """特徴点から各指標を計算して Metrics を返す"""
        m = Metrics()
        pose = None

        if lm is not None:
            m.face_detected = True
            self._last_face_time = now

            # ---- 1. EAR とまばたき ----
            ear = (eye_aspect_ratio(lm[RIGHT_EYE_EAR]) + eye_aspect_ratio(lm[LEFT_EYE_EAR])) / 2.0
            m.ear = ear
            self._update_blink(ear, now)

            # ---- 2. 目の開き具合（EAR の移動平均）----
            self._ear_hist.append((now, ear))

            # ---- 3. 顔の向き（solvePnP）----
            pose = self._estimate_head_pose(lm, shape)
            if pose is not None:
                yaw, pitch = rotation_to_yaw_pitch(pose[0])
                if self._calibrate_request:
                    self._yaw0, self._pitch0 = yaw, pitch
                    self._calibrate_request = False
                m.yaw = yaw - self._yaw0
                m.pitch = pitch - self._pitch0
                m.head_angle = math.hypot(m.yaw, m.pitch)

            # ---- 4. 視線（虹彩中心の相対位置）----
            hr, vr = iris_relative_position(lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]],
                                            lm[RIGHT_IRIS_CENTER])
            hl, vl = iris_relative_position(lm[LEFT_EYE_CORNERS[0]], lm[LEFT_EYE_CORNERS[1]],
                                            lm[LEFT_IRIS_CENTER])
            # まばたき中は虹彩が隠れて値が乱れるので記録しない
            if ear >= config.EAR_BLINK_THRESHOLD:
                self._gaze_hist.append((now, (hr + hl) / 2.0, (vr + vl) / 2.0))

        # 古い履歴を捨てる
        while self._ear_hist and now - self._ear_hist[0][0] > config.EAR_AVG_SEC:
            self._ear_hist.popleft()
        while self._gaze_hist and now - self._gaze_hist[0][0] > config.GAZE_WINDOW_SEC:
            self._gaze_hist.popleft()
        while self._blink_times and now - self._blink_times[0] > config.BLINK_WINDOW_SEC:
            self._blink_times.popleft()

        if self._ear_hist:
            m.ear_avg = float(np.mean([e for _, e in self._ear_hist]))

        if len(self._gaze_hist) >= 10:
            arr = np.array([(h, v) for _, h, v in self._gaze_hist])
            # 標準偏差 = sqrt(横方向の分散 + 縦方向の分散)
            m.gaze_std = float(math.sqrt(arr[:, 0].var() + arr[:, 1].var()))

        # まばたき回数/分：起動直後は経過時間で割って推定し、十分たつまでは未計算とする
        elapsed = now - self._start_time
        if elapsed >= config.BLINK_WARMUP_SEC:
            window = min(elapsed, config.BLINK_WINDOW_SEC)
            m.blink_rate = len(self._blink_times) * 60.0 / window
        m.blink_count = self._blink_count

        # ---- 5. 離席判定 ----
        m.absent = (self._last_face_time is None
                    or now - self._last_face_time > config.ABSENT_GRACE_SEC)
        return m, pose

    def _update_blink(self, ear, now):
        """
        まばたきの検出。EAR がしきい値を下回った時刻を覚えておき、
        再び上回ったとき、閉じていた時間が適切な範囲なら 1 回と数える。
        """
        if ear < config.EAR_BLINK_THRESHOLD:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
        else:
            if self._eye_closed_since is not None:
                duration = now - self._eye_closed_since
                if config.BLINK_MIN_DURATION <= duration <= config.BLINK_MAX_DURATION:
                    self._blink_times.append(now)
                    self._blink_count += 1
                self._eye_closed_since = None

    def _estimate_head_pose(self, lm, shape):
        """
        cv2.solvePnP で頭部姿勢を推定する。
        カメラの内部パラメータは、焦点距離 = 画像の幅、光学中心 = 画像中心と近似する。
        戻り値: (rvec, tvec, camera_matrix)、失敗時は None
        """
        h, w = shape[:2]
        image_points = lm[POSE_LANDMARKS]
        camera_matrix = np.array([[w, 0, w / 2.0],
                                  [0, w, h / 2.0],
                                  [0, 0, 1.0]], dtype=np.float64)
        dist = np.zeros((4, 1))
        ok, rvec, tvec = cv2.solvePnP(FACE_MODEL_3D, image_points, camera_matrix, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        return rvec, tvec, camera_matrix

    # --------------------------------------------------------
    # デバッグ表示
    # --------------------------------------------------------
    def _draw_debug(self, frame, lm, m, focus, pose):
        """カメラ映像に特徴点・顔の向き・各指標を描き込んだ画像を作る"""
        img = frame.copy()
        if lm is not None:
            # 目の EAR 用の点（緑）と虹彩中心（赤）
            for idx in RIGHT_EYE_EAR + LEFT_EYE_EAR:
                cv2.circle(img, tuple(lm[idx].astype(int)), 2, (0, 255, 0), -1)
            for idx in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                cv2.circle(img, tuple(lm[idx].astype(int)), 3, (0, 0, 255), -1)
            # 頭部姿勢推定に使った点（黄）
            for idx in POSE_LANDMARKS:
                cv2.circle(img, tuple(lm[idx].astype(int)), 3, (0, 255, 255), -1)
            # 顔の向きを鼻先から伸びる線で表示
            if pose is not None:
                rvec, tvec, cam = pose
                tip, _ = cv2.projectPoints(np.array([[0.0, 0.0, -400.0]]), rvec, tvec, cam,
                                           np.zeros((4, 1)))
                p0 = tuple(lm[1].astype(int))
                p1 = tuple(tip.ravel().astype(int))
                cv2.line(img, p0, p1, (255, 128, 0), 2)

        def fmt(v, f="{:.2f}"):
            return "--" if v is None else f.format(v)

        lines = [
            f"FOCUS {focus.focus:5.1f}  (raw {focus.raw:5.1f})",
            "ABSENT" if m.absent else ("FACE OK" if m.face_detected else "face lost"),
            f"EAR {fmt(m.ear)}  avg {fmt(m.ear_avg)}",
            f"blink {m.blink_count}  rate {fmt(m.blink_rate, '{:.1f}')}/min",
            f"yaw {fmt(m.yaw, '{:+.1f}')}  pitch {fmt(m.pitch, '{:+.1f}')}  dev {fmt(m.head_angle, '{:.1f}')}",
            f"gaze std {fmt(m.gaze_std, '{:.3f}')}",
            f"camera {m.camera_fps:4.1f} fps",
        ]
        for i, text in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img
