"""
main.py
-------
作品のエントリーポイント。
  1. カメラ処理スレッド（camera_tracker.py）を起動する。カメラがなければテストモードで起動
  2. メインスレッドで描画ループ（renderer.py）を回す
  3. キー操作を処理し、デバッグ用カメラウィンドウ（cv2.imshow）を表示する
  4. 終了時に集中度の推移を CSV に保存する

使い方:
    python main.py              # カメラを使って起動
    python main.py --no-camera  # カメラを使わずテストモードで起動
    python main.py --camera 1   # 別のカメラ番号を使う

キー操作（3D ウィンドウ上で）:
    ESC : 終了してログを保存
    D   : デバッグ用カメラウィンドウの表示切り替え
    R   : 成長状態をリセット
    1/2 : 押している間、集中度を下げる / 上げる（テストモード）
    0   : テストモードをやめてカメラによる推定に戻る
    C   : 今の顔の向きを「正面」として登録する（カメラが画面の横や上にある場合用）
"""

import argparse
import os
import sys
import time

# Wayland 環境（Ubuntu の標準）でも安定して動くよう、ウィンドウは X11 互換モード（XWayland）で開く
#   QT_QPA_PLATFORM        : OpenCV のウィンドウ（Qt）用
#   PYGLFW_LIBRARY_VARIANT : glfw 用（Wayland 版は GNOME の設定と合わず異常終了することがある）
#   PYOPENGL_PLATFORM      : PyOpenGL 用（X11 版 glfw に合わせて GLX を使う）
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import cv2      # noqa: E402
import glfw     # noqa: E402

import config   # noqa: E402
from camera_tracker import CameraTracker   # noqa: E402
from renderer import Renderer               # noqa: E402
from session_logger import SessionLogger    # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_WINDOW = "Camera Debug"


class App:
    """アプリ全体の状態（モード・キー入力・各モジュール）をまとめるクラス"""

    def __init__(self, args):
        # ---- カメラ ----
        self.tracker = None
        if not args.no_camera:
            tracker = CameraTracker(args.camera)
            if tracker.open():
                self.tracker = tracker
                self.tracker.start()
                print(f"[main] カメラ {args.camera} を開きました")
            else:
                print(f"[main] カメラ {args.camera} を開けませんでした → テストモードで起動します",
                      file=sys.stderr)

        # カメラがないときは最初からテストモード
        self.manual_mode = self.tracker is None
        self.manual_focus = 50.0
        self.debug_on = self.tracker is not None and not args.no_debug
        if self.tracker is not None:
            self.tracker.debug_enabled = self.debug_on

        # ---- 描画 ----
        self.renderer = Renderer()
        glfw.set_key_callback(self.renderer.window, self.on_key)

        self.logger = SessionLogger(BASE_DIR)

    # --------------------------------------------------------
    # キー入力
    # --------------------------------------------------------
    def on_key(self, window, key, scancode, action, mods):
        """押した瞬間に 1 回だけ実行したい操作（1/2 の長押しはループ内で処理する）"""
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_D:
            self.toggle_debug()
        elif key == glfw.KEY_R:
            self.reset_growth()
        elif key == glfw.KEY_0:
            if self.tracker is not None:
                self.manual_mode = False
                print("[main] カメラによる推定に戻りました")
        elif key == glfw.KEY_C:
            if self.tracker is not None:
                self.tracker.request_calibration()
                print("[main] 今の顔の向きを正面として登録しました")

    def toggle_debug(self):
        if self.tracker is None:
            print("[main] カメラがないのでデバッグウィンドウは使えません")
            return
        self.debug_on = not self.debug_on
        self.tracker.debug_enabled = self.debug_on
        if not self.debug_on:
            cv2.destroyWindow(DEBUG_WINDOW)
            cv2.waitKey(1)

    def reset_growth(self):
        # 成長の表現は次の段階で追加する
        print("[main] 成長状態をリセットしました")

    def handle_held_keys(self, dt):
        """1/2 キーを押している間、集中度を連続的に変える"""
        win = self.renderer.window
        down = glfw.get_key(win, glfw.KEY_1) == glfw.PRESS
        up = glfw.get_key(win, glfw.KEY_2) == glfw.PRESS
        if down or up:
            if not self.manual_mode:
                # カメラモードから切り替えるときは、今の値から始める
                if self.tracker is not None:
                    self.manual_focus = self.tracker.get_state()[1].focus
                self.manual_mode = True
            delta = config.MANUAL_FOCUS_SPEED * dt * (1 if up else -1)
            self.manual_focus = max(0.0, min(100.0, self.manual_focus + delta))

    # --------------------------------------------------------
    # デバッグウィンドウ（OpenCV）
    # --------------------------------------------------------
    def update_debug_window(self):
        if not self.debug_on or self.tracker is None:
            return
        frame = self.tracker.pop_debug_frame()
        if frame is not None:
            cv2.imshow(DEBUG_WINDOW, frame)
        # waitKey を呼ばないと OpenCV のウィンドウが更新されない
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            glfw.set_window_should_close(self.renderer.window, True)
        elif key in (ord("d"), ord("D")):
            self.toggle_debug()

    # --------------------------------------------------------
    # メインループ
    # --------------------------------------------------------
    def run(self):
        start = time.monotonic()
        prev = start
        fps = 0.0
        try:
            while not self.renderer.should_close():
                now = time.monotonic()
                dt = min(now - prev, 0.1)      # 一時停止後などに大きく飛ばないよう上限を設ける
                prev = now
                if dt > 0:
                    fps = 0.95 * fps + 0.05 / dt if fps > 0 else 1.0 / dt

                glfw.poll_events()
                self.handle_held_keys(dt)

                # ---- 集中度を決める ----
                metrics, scores, raw = None, None, None
                if self.tracker is not None:
                    metrics, fstate = self.tracker.get_state()
                    scores, raw = fstate.scores, fstate.raw
                if self.manual_mode:
                    focus = self.manual_focus
                    raw = self.manual_focus
                    mode = "TEST MODE (1/2 to change" + (", 0 = camera)" if self.tracker else ")")
                    mode_color = (255, 200, 110, 255)
                    if self.tracker is None:
                        mode = "NO CAMERA - " + mode
                else:
                    focus = fstate.focus
                    mode = "CAMERA MODE"
                    mode_color = (140, 230, 160, 255)

                info = {
                    "focus": focus, "mode": mode, "mode_color": mode_color,
                    "metrics": metrics, "scores": scores, "render_fps": fps,
                }
                self.renderer.render(focus, info, dt)
                self.update_debug_window()

                self.logger.record(now - start, "test" if self.manual_mode else "camera",
                                   focus, raw, metrics, scores)
        finally:
            self.shutdown()

    def shutdown(self):
        if self.tracker is not None:
            self.tracker.stop()
        cv2.destroyAllWindows()
        self.renderer.close()
        path = self.logger.save()
        print(f"[main] ログを保存しました: {path}")


def main():
    parser = argparse.ArgumentParser(description="集中度で光る・育つ 3D の脳")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="カメラ番号")
    parser.add_argument("--no-camera", action="store_true", help="カメラを使わずテストモードで起動")
    parser.add_argument("--no-debug", action="store_true", help="デバッグウィンドウを最初は閉じておく")
    args = parser.parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
