#version 330 core
// 3D の場面（机・物・カゴ・ロボット・影）の頂点シェーダー。部品ごとに uModel を変えて同じメッシュを使い回す
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;     // 部品の変換（拡大縮小を含む）
uniform mat4 uRot;       // 部品の回転だけ（法線の変換用。立方体・円柱は軸方向の拡大なので、これで向きは正しい）
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vWorld;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorld = world.xyz;
    vNormal = mat3(uRot) * aNormal;
    gl_Position = uProj * uView * world;
}
