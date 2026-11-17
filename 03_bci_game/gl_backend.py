"""
gl_backend.py
-------------
OpenGL の描画方式（GPU か、ソフトウェア描画か）を選ぶ。glfw・PyOpenGL・cv2 を import する前に setup() を呼ぶ。
（環境変数でライブラリの種類が決まるので、import したあとでは変えられない）

■ 方式
  "gpu"      : Wayland のネイティブのウィンドウ + EGL。Intel などの GPU で描く
  "software" : X11 互換モード（XWayland）のウィンドウ + GLX。この PC では XWayland が DRI3 に対応していないため、
               CPU で描くソフトウェア描画（llvmpipe）になる（以前からの方式。どの環境でもほぼ確実に動く）

■ 選び方
  1. --software が指定されている、または Wayland のセッションでない → "software"
  2. 小さなテスト（別のプロセス）で、Wayland + EGL のウィンドウを作って GL_RENDERER を調べる。
     起動でき、ソフトウェア描画（llvmpipe など）でなければ "gpu"。落ちた・時間切れ・ソフトウェア描画なら "software"
     （別のプロセスで試すので、テストが異常終了してもゲームは落ちない。かかる時間は 1 秒くらい）
  3. 環境変数 PYGLFW_LIBRARY_VARIANT がすでに設定されていれば、それに従う（自分で方式を決めたいとき）

■ Wayland で glfw が落ちる問題
  glfw（Wayland 版）が窓枠を描くのに使う libdecor の GTK プラグインが、GNOME の設定（GSettings）のキーが
  見つからずに異常終了する（Settings schema ... does not contain a key named 'antialiasing'）。
  GTK プラグインを使わないよう、cairo プラグインだけを入れたフォルダを作って LIBDECOR_PLUGIN_DIR で指定する。
  cairo プラグイン（Ubuntu の libdecor-0-plugin-1-cairo）が入っていなければ空のフォルダになり、窓枠なしで開く
  （ESC で終了、F11 で全画面。ウィンドウの移動・大きさの変更はできない）。

■ 垂直同期を使わない（GPU のとき）
  Wayland では、ウィンドウが見えていない（ほかのワークスペース・画面ロックなど）あいだ、垂直同期つきの
  swap_buffers が返ってこなくなり、ゲームが止まることがある。GPU のときは swap_interval(0) にして、
  FrameLimiter で FRAME_CAP_FPS（60）に抑える。ソフトウェア描画は 60 fps に届かないので、今までどおり垂直同期を使う。
"""

import json
import os
import subprocess
import sys
import time

SOFT_NAMES = ("llvmpipe", "softpipe", "swrast", "software")
SYS_LIBDECOR = "/usr/lib/x86_64-linux-gnu/libdecor/plugins-1"
PROBE_TIMEOUT = 8.0

backend = None          # 選んだ方式（"gpu" / "software"）
reason = ""             # 選んだ理由（起動時に表示する）

_PROBE = r"""
import json, glfw
r = {}
if glfw.init():
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
    glfw.window_hint(glfw.VISIBLE, False)
    w = glfw.create_window(64, 64, "probe", None, None)
    if w:
        glfw.make_context_current(w)
        from OpenGL import GL as gl
        r["renderer"] = gl.glGetString(gl.GL_RENDERER).decode()
        glfw.destroy_window(w)
    glfw.terminate()
print("PROBE " + json.dumps(r))
"""


def _libdecor_dir():
    """cairo プラグインだけを入れたフォルダ（~/.cache の下）。cairo がなければ空のフォルダ（窓枠なし）"""
    d = os.path.join(os.path.expanduser("~"), ".cache", "bci_operator", "libdecor-plugins")
    try:
        os.makedirs(d, exist_ok=True)
        src = os.path.join(SYS_LIBDECOR, "libdecor-cairo.so")
        dst = os.path.join(d, "libdecor-cairo.so")
        if os.path.lexists(dst) and not os.path.exists(dst):
            os.remove(dst)                       # リンク先がなくなった古いリンクは消す
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    except OSError:
        pass
    return d


def _gpu_env():
    return {"PYGLFW_LIBRARY_VARIANT": "wayland", "PYOPENGL_PLATFORM": "egl",
            "LIBDECOR_PLUGIN_DIR": os.environ.get("LIBDECOR_PLUGIN_DIR") or _libdecor_dir()}


def _probe(env):
    """別のプロセスで GL_RENDERER を調べる。(renderer または None, 失敗の理由)"""
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE], env={**os.environ, **env}, capture_output=True,
                             text=True, timeout=PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"テストが起動しない・時間切れ（{type(e).__name__}）"
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            r = json.loads(line[6:]).get("renderer")
            return r, ("" if r else "ウィンドウを作れない")
    return None, f"テストが異常終了（終了コード {out.returncode}）"


def _set(env):
    for k, v in env.items():
        os.environ[k] = v


def setup(force_software=False):
    """描画方式を選んで環境変数を設定する。選んだ方式（"gpu" / "software"）を返す"""
    global backend, reason
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")       # OpenCV のウィンドウ（Qt）は X11 互換モードで
    soft = {"PYGLFW_LIBRARY_VARIANT": "x11", "PYOPENGL_PLATFORM": "glx"}
    if "PYGLFW_LIBRARY_VARIANT" in os.environ:
        backend = "gpu" if os.environ["PYGLFW_LIBRARY_VARIANT"] == "wayland" else "software"
        if backend == "gpu":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            os.environ.setdefault("LIBDECOR_PLUGIN_DIR", _libdecor_dir())
        else:
            os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
        reason = "環境変数 PYGLFW_LIBRARY_VARIANT の指定"
        return backend
    if force_software:
        backend, reason = "software", "--software の指定"
    elif not os.environ.get("WAYLAND_DISPLAY"):
        backend, reason = "software", "Wayland のセッションではない"
    else:
        env = _gpu_env()
        renderer, why = _probe(env)
        if renderer is not None and not any(s in renderer.lower() for s in SOFT_NAMES):
            _set(env)
            backend, reason = "gpu", f"GPU が使える（{renderer}）"
            return backend
        backend = "software"
        reason = f"GPU が使えない：{why or renderer}"
    _set(soft)
    return backend


def swap_interval():
    """glfw.swap_interval に渡す値：GPU は 0（FrameLimiter で抑える）、ソフトウェア描画は 1（垂直同期）"""
    return 0 if backend == "gpu" else 1


class FrameLimiter:
    """1 フレームが 1/fps 秒より短ければ、残りを待つ（GPU のときだけ。垂直同期の代わり）"""

    def __init__(self, fps):
        self.period = 1.0 / fps if (fps and backend == "gpu") else 0.0
        self.next = time.perf_counter()

    def wait(self):
        if self.period <= 0:
            return
        now = time.perf_counter()
        if self.next > now:
            time.sleep(self.next - now)
            self.next += self.period
        else:
            self.next = now + self.period         # 遅れているときは待たない（遅れを取り返そうとしない）
