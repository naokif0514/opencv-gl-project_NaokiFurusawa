#version 330 core
// ロボットアームの部品（円柱・球）の頂点シェーダー
// 部品ごとに uModel（位置・向き・大きさ）を変えて、同じメッシュを使い回す。
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;     // 部品の変換（拡大縮小を含む）
uniform mat4 uRot;       // 部品の回転だけ（法線の変換用。円柱は太さ方向の拡大が等しいのでこれで正しい）
uniform mat4 uProj;      // 正射影（右の平面の座標 0〜1 をそのまま画面に対応させる）

out vec3 vNormal;
out vec3 vWorld;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorld = world.xyz;
    vNormal = mat3(uRot) * aNormal;
    gl_Position = uProj * world;
}
