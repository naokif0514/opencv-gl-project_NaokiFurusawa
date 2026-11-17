"""
gl_backend.py
-------------
OpenGL の描画方式（GPU か、ソフトウェア描画か）を選ぶ。glfw・PyOpenGL・cv2 を import する前に setup() を呼び、
glfw.init() の代わりに init_glfw(glfw) を使う。
（環境変数と sys.path でライブラリが決まるので、import したあとでは変えられない）

■ 方式
  "gpu"      : Wayland のネイティブのウィンドウ + EGL。Intel などの GPU で描く。
               03 専用に入れた新しい glfw（.libs/、pyglfw 2.11.0 = GLFW 3.4）を使う
  "software" : X11 互換モード（XWayland）のウィンドウ + GLX。この PC では XWayland が DRI3 に対応していないため、
               CPU で描くソフトウェア描画（llvmpipe）になる（以前からの方式。.venv の glfw 2.7.0 を使う）

■ 選び方
  1. --software が指定されている、または Wayland のセッションでない → "software"
  2. 小さなテスト（別のプロセス）で、Wayland + EGL のウィンドウを作って GL_RENDERER と GLFW の版を調べる。
     GLFW が 3.4 以上で、ソフトウェア描画（llvmpipe など）でなければ "gpu"。それ以外は "software"
     （別のプロセスで試すので、テストが異常終了してもゲームは落ちない。かかる時間は 0.3 秒くらい）
  3. 環境変数 PYGLFW_LIBRARY_VARIANT がすでに設定されていれば、それに従う（自分で方式を決めたいとき）

■ GLFW 3.4 が必要な理由（実際に確認したこと）
  .venv の pyglfw 2.7.0 に入っている GLFW 3.3.9 の Wayland 版は、GNOME 46 でウィンドウにクリックもキーも
  届かなかった（タイトル画面から進めない）。GLFW 3.4 では届く。01・02 に影響しないよう、.venv は変えずに
  03 専用のフォルダへ入れる：
      .venv/bin/pip install --no-deps --target 03_bci_game/.libs glfw==2.11.0

■ 窓枠（libdecor を使わない）
  Wayland 版の glfw が窓枠に使う libdecor の GTK プラグインは、GNOME の設定（GSettings）のキーが見つからずに
  異常終了する（Settings schema ... does not contain a key named 'antialiasing'）。GLFW 3.4 の
  GLFW_WAYLAND_LIBDECOR = DISABLE で libdecor を使わず、GLFW が自分で描く簡単な枠にする。

■ 垂直同期を使わない（GPU のとき）
  Wayland では、ウィンドウが見えていない（最小化・ほかのワークスペースなど）あいだ、垂直同期つきの
  swap_buffers が返ってこず、ゲームが止まる（実際に確認）。GPU のときは swap_interval(0) にして、
  FrameLimiter で FRAME_CAP_FPS（60）に抑える。ソフトウェア描画は 60 fps に届かないので、今までどおり垂直同期を使う。
"""

import json
import os
import subprocess
import sys
import time

LIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".libs")
SOFT_NAMES = ("llvmpipe", "softpipe", "swrast", "software")
MIN_GLFW = (3, 4)
PROBE_TIMEOUT = 8.0

backend = None          # 選んだ方式（"gpu" / "software"）
reason = ""             # 選んだ理由（起動時に表示する）

_PROBE = r"""
import json, glfw
r = {"glfw": glfw.get_version_string().decode()}
if hasattr(glfw, "WAYLAND_LIBDECOR"):
    glfw.init_hint(glfw.WAYLAND_LIBDECOR, glfw.WAYLAND_DISABLE_LIBDECOR)
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


def _gpu_env():
    pp = os.environ.get("PYTHONPATH")
    return {"PYGLFW_LIBRARY_VARIANT": "wayland", "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": LIBS + (os.pathsep + pp if pp else "")}


def _version(s):
    try:
        return tuple(int(x) for x in s.split()[0].split(".")[:2])
    except (ValueError, IndexError):
        return (0, 0)


def _probe(env):
    """別のプロセスで調べる。(結果の dict または None, 失敗の理由)"""
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE], env={**os.environ, **env}, capture_output=True,
                             text=True, timeout=PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"テストが起動しない・時間切れ（{type(e).__name__}）"
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:]), ""
    return None, f"テストが異常終了（終了コード {out.returncode}）"


def _use_gpu():
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "wayland"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if os.path.isdir(LIBS) and LIBS not in sys.path:
        sys.path.insert(0, LIBS)              # 03 専用の新しい glfw を先に探す


def setup(force_software=False):
    """描画方式を選んで環境変数・sys.path を設定する。選んだ方式（"gpu" / "software"）を返す"""
    global backend, reason
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")       # OpenCV のウィンドウ（Qt）は X11 互換モードで
    if "PYGLFW_LIBRARY_VARIANT" in os.environ:
        if os.environ["PYGLFW_LIBRARY_VARIANT"] == "wayland":
            backend = "gpu"
            _use_gpu()
        else:
            backend = "software"
            os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
        reason = "環境変数 PYGLFW_LIBRARY_VARIANT の指定"
        return backend
    if force_software:
        backend, reason = "software", "--software の指定"
    elif not os.environ.get("WAYLAND_DISPLAY"):
        backend, reason = "software", "Wayland のセッションではない"
    else:
        r, why = _probe(_gpu_env())
        renderer = (r or {}).get("renderer")
        ver = (r or {}).get("glfw", "")
        if r is None:
            reason = f"GPU が使えない：{why}"
        elif _version(ver) < MIN_GLFW:
            reason = (f"GLFW {ver.split()[0]} は Wayland で入力が届かない（03_bci_game/.libs に glfw==2.11.0 を"
                      f"入れると GPU を使える。README 8 章）")
        elif renderer is None:
            reason = "GPU が使えない：ウィンドウを作れない"
        elif any(s in renderer.lower() for s in SOFT_NAMES):
            reason = f"GPU が使えない：{renderer}"
        else:
            _use_gpu()
            backend, reason = "gpu", f"GPU が使える（{renderer}、GLFW {ver.split()[0]}）"
            return backend
        backend = "software"
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    return backend


def init_glfw(glfw):
    """glfw.init() の代わり。GPU（Wayland）のときは libdecor を使わない（GTK プラグインで落ちるため）"""
    if backend == "gpu" and hasattr(glfw, "WAYLAND_LIBDECOR"):
        glfw.init_hint(glfw.WAYLAND_LIBDECOR, glfw.WAYLAND_DISABLE_LIBDECOR)
    return glfw.init()


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
