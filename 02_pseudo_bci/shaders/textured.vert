#version 330 core
// 画像（テクスチャ）を画面上の四角形に貼るための頂点シェーダー
// 文字の重ね表示（OpenCV で描いた画像）と、ラスタープロットに使う。
layout(location = 0) in vec2 aPos;   // 0〜1 の四角形
uniform vec2 uScreen;                // 画面サイズ [画素]
uniform vec4 uRect;                  // 貼る場所 (x, y, 幅, 高さ) [画素, 左上原点]
out vec2 vUV;
void main() {
    vec2 p = uRect.xy + aPos * uRect.zw;
    vUV = aPos;                      // 画像の 1 行目（上端）が v=0
    gl_Position = vec4(p.x / uScreen.x * 2.0 - 1.0, 1.0 - p.y / uScreen.y * 2.0, 0.0, 1.0);
}
