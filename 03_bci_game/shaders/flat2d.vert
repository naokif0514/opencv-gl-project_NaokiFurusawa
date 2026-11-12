#version 330 core
// （コピー元：02_pseudo_bci/shaders/flat2d.vert）
// 2D 図形（円・線・四角）の頂点シェーダー
// 画面の画素座標（左上が原点）を、正規化デバイス座標に変換する。
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
uniform vec2 uScreen;
out vec4 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos.x / uScreen.x * 2.0 - 1.0, 1.0 - aPos.y / uScreen.y * 2.0, 0.0, 1.0);
}
