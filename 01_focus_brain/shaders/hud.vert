#version 330 core
// HUD（画面の隅に重ねる情報パネル）の頂点シェーダー
// 画素単位の座標を、画面全体を -1〜1 とする正規化デバイス座標に変換する。

layout(location = 0) in vec2 aPos;   // 0〜1 の四角形
uniform vec2 uScreen;                // 画面サイズ [画素]
uniform vec4 uRect;                  // パネルの位置とサイズ (x, y, 幅, 高さ) [画素, 左上原点]

out vec2 vUV;

void main() {
    vec2 pixel = uRect.xy + aPos * uRect.zw;
    vec2 ndc = vec2(pixel.x / uScreen.x * 2.0 - 1.0, 1.0 - pixel.y / uScreen.y * 2.0);
    // 画像の 1 行目（上端）がテクスチャの v=0 に入っているので、そのまま対応させる
    vUV = aPos;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
