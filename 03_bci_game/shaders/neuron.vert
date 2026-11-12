#version 330 core
// （コピー元：02_pseudo_bci/shaders/neuron.vert）
// ニューロン（小さな球）の頂点シェーダー
// 1 つの球のメッシュを、インスタンス描画で N 個のニューロンの位置に配置する。

layout(location = 0) in vec3 aPos;      // 単位球の頂点（法線も同じ向き）
layout(location = 1) in vec3 iOffset;   // ニューロンの位置（インスタンスごと）
layout(location = 2) in vec3 iColor;    // 好きな方向に対応した色（インスタンスごと）
layout(location = 3) in float iFlash;   // スパイクによる光り具合 0〜1（インスタンスごと）
layout(location = 4) in float iAlive;   // 1: 記録できている, 0: 脱落
layout(location = 5) in float iActivity; // 発火率がベースラインよりどれだけ高いか 0〜1

uniform mat4 uView;
uniform mat4 uProj;
uniform float uRadius;

out vec3 vNormal;
out vec3 vWorld;
out vec3 vColor;
out float vFlash;
out float vAlive;
out float vActivity;

void main() {
    // よく発火しているほど、また光った瞬間は少し大きくする
    float r = uRadius * (1.0 + (0.30 * iActivity + 0.12 * iFlash) * iAlive);
    vec3 world = iOffset + aPos * r;
    vNormal = aPos;
    vWorld = world;
    vColor = iColor;
    vFlash = iFlash;
    vAlive = iAlive;
    vActivity = iActivity;
    gl_Position = uProj * uView * vec4(world, 1.0);
}
