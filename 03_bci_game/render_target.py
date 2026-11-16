"""
render_target.py
----------------
ゲームの画面を、決まった解像度のオフスクリーン（フレームバッファオブジェクト、FBO）に描き、
最後にウィンドウ（ディスプレイいっぱい）へ拡大して表示する。

  ・ソフトウェア描画（llvmpipe）では、描く画素の数がそのまま重さになる。ウィンドウを大きくしても、
    描く解像度（RENDER_SCALE × 1280×720）は変えずに、最後の拡大（glBlitFramebuffer、線形補間）だけを大きくする
  ・縦横比（16:9）を保ち、余った部分は黒い帯にする（レターボックス）
  ・MSAA_SAMPLES > 0 のときは、マルチサンプルの FBO に描いてから、同じ大きさの普通の FBO にまとめ（resolve）、
    それを拡大する（マルチサンプルの FBO からは、大きさを変えながら直接コピーできないため）
"""

import glfw
from OpenGL import GL as gl


class RenderTarget:
    def __init__(self, width, height, samples=0):
        self.width, self.height, self.samples = int(width), int(height), int(samples)
        self.fbo, self.color = self._make(self.samples)
        self.resolve = None
        if self.samples > 0:
            self.resolve, _ = self._make(0)

    def _make(self, samples):
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        color = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color)
        if samples > 0:
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, gl.GL_RGBA8, self.width, self.height)
        else:
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color)
        depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
        if samples > 0:
            gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, gl.GL_DEPTH24_STENCIL8,
                                                self.width, self.height)
        else:
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, gl.GL_RENDERBUFFER, depth)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"フレームバッファを作れませんでした（状態 {status}）")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return fbo, color

    def bind(self):
        """これから描くものを、オフスクリーンに描くようにする"""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)

    def letterbox(self, out_w, out_h):
        """(出力先の x, y, 幅, 高さ)：縦横比を保って、出力先の中央にできるだけ大きく置く"""
        s = min(out_w / self.width, out_h / self.height)
        w, h = int(self.width * s), int(self.height * s)
        return (out_w - w) // 2, (out_h - h) // 2, w, h

    def present(self, window):
        """オフスクリーンの画面を、ウィンドウいっぱいに拡大して表示する（縦横比は保つ）"""
        src = self.fbo
        if self.resolve is not None:            # マルチサンプルをまとめる（同じ大きさでコピー）
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.resolve)
            gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                                 gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
            src = self.resolve
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        x, y, w, h = self.letterbox(fb_w, fb_h)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glBlitFramebuffer(0, 0, self.width, self.height, x, y, x + w, y + h,
                             gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
