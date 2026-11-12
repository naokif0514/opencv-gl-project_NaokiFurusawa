"""
hand_tracker.py
---------------
（コピー元：02_pseudo_bci/hand_tracker.py。03_bci_game で流用するためにコピーした）
Web カメラ（OpenCV）で手を撮影し、MediaPipe Hands で手の特徴点を検出して、
手の位置と速度ベクトルを求める。これがシミュレーションへの「本当の運動」の入力になる。

・位置：config.HAND_LANDMARK 番の特徴点（初期値は人差し指の付け根）を使う。
        画面の幅・高さを 1 とした座標で、x は右向き、y は上向き。
        カメラ映像は左右を反転（鏡像）して、自分から見た向きと画面の向きをそろえる。
・速度：前のフレームとの位置の差 ÷ 時間。手ぶれを抑えるため、直近 VELOCITY_SMOOTH_SEC 秒の
        移動平均をとる。
・手が見えないとき：速度 0 として扱う。

カメラ処理は別スレッドで動かし、描画とシミュレーションのフレームレートに影響しないようにしている。
テストモード用に、マウスを手の代わりに使う MouseHand も用意している（同じ形で値を返す）。
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

import config


@dataclass
class HandState:
    """手の状態（描画・シミュレーション側に渡す値）"""
    detected: bool = False
    pos: np.ndarray | None = None          # 位置 (x, y)、0〜1
    vel: np.ndarray = None                 # 速度 (vx, vy) [画面/秒]
    fps: float = 0.0

    def __post_init__(self):
        if self.vel is None:
            self.vel = np.zeros(2)


class VelocityEstimator:
    """位置の時系列から、移動平均でなめらかにした速度を求める"""

    def __init__(self):
        self.samples = deque()     # (時刻, 瞬間速度)
        self.prev = None           # (時刻, 位置)

    def reset(self):
        self.samples.clear()
        self.prev = None

    def update(self, pos, now):
        """新しい位置を受け取り、なめらかにした速度を返す。pos が None なら速度 0"""
        if pos is None:
            self.reset()
            return np.zeros(2)
        if self.prev is not None:
            dt = now - self.prev[0]
            if dt > 1e-4:
                v = (pos - self.prev[1]) / dt
                self.samples.append((now, v))
        self.prev = (now, pos.copy())
        # 古いサンプルを捨てて、残りを平均する（移動平均）
        while self.samples and now - self.samples[0][0] > config.VELOCITY_SMOOTH_SEC:
            self.samples.popleft()
        if not self.samples:
            return np.zeros(2)
        v = np.mean([s[1] for s in self.samples], axis=0)
        # 極端に速い値（検出の飛び）は上限で切る
        speed = np.linalg.norm(v)
        if speed > config.SPEED_CLIP:
            v = v * (config.SPEED_CLIP / speed)
        return v


class HandTracker(threading.Thread):
    """カメラ + MediaPipe Hands による手の追跡スレッド"""

    def __init__(self, camera_index=config.CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = None
        self._lock = threading.Lock()
        self._state = HandState()
        self._debug_frame = None
        self._running = False
        self.debug_enabled = True
        self.velocity = VelocityEstimator()

    def open(self):
        """カメラを開く。開けなければ False（テストモードで起動するため）"""
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

    def pop_debug_frame(self):
        with self._lock:
            f = self._debug_frame
            self._debug_frame = None
            return f

    def run(self):
        self._running = True
        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        drawing = mp.solutions.drawing_utils
        prev_t = time.monotonic()
        last_seen = -1e9
        fps = 0.0

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)       # 鏡像にする（自分が右に動かすと画面でも右へ）
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pos = None
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark[config.HAND_LANDMARK]
                pos = np.array([lm.x, 1.0 - lm.y])   # y を上向きに
                last_seen = now

            detected = pos is not None
            if detected:
                vel = self.velocity.update(pos, now)
            elif now - last_seen > config.HAND_LOST_SEC:
                vel = self.velocity.update(None, now)   # 見えなくなったら速度 0
            else:
                vel = self._state.vel                    # 一瞬の見失いは直前の値を保つ

            state = HandState(detected=detected, pos=pos, vel=vel, fps=fps)

            debug = None
            if self.debug_enabled:
                debug = frame.copy()
                if result.multi_hand_landmarks:
                    drawing.draw_landmarks(debug, result.multi_hand_landmarks[0],
                                           mp.solutions.hands.HAND_CONNECTIONS)
                    h, w = debug.shape[:2]
                    px = (int(pos[0] * w), int((1.0 - pos[1]) * h))
                    cv2.circle(debug, px, 9, (0, 255, 255), 2)
                    # 速度ベクトルを矢印で表示（0.3 秒分の移動量）
                    tip = (int(px[0] + vel[0] * w * 0.3), int(px[1] - vel[1] * h * 0.3))
                    cv2.arrowedLine(debug, px, tip, (0, 200, 255), 2, tipLength=0.25)
                text = (f"hand {'OK' if detected else 'NOT FOUND'}  "
                        f"v=({vel[0]:+.2f},{vel[1]:+.2f})  {fps:4.1f} fps")
                cv2.putText(debug, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(debug, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255) if detected else (80, 80, 255), 1, cv2.LINE_AA)

            with self._lock:
                self._state = state
                if debug is not None:
                    self._debug_frame = debug

        hands.close()


class MouseHand:
    """テストモード用：マウスカーソルの位置を「手」として扱う"""

    def __init__(self):
        self.velocity = VelocityEstimator()
        self.state = HandState()

    def update(self, x_norm, y_norm, inside, now):
        """x_norm, y_norm: ウィンドウ内の位置（0〜1, y は上向き）、inside: ウィンドウ内か"""
        if not inside:
            self.state = HandState(detected=False, pos=None, vel=self.velocity.update(None, now))
            return self.state
        pos = np.array([x_norm, y_norm])
        self.state = HandState(detected=True, pos=pos, vel=self.velocity.update(pos, now))
        return self.state
