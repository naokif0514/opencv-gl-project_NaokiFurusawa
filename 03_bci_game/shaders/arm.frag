#version 330 core
// （コピー元：02_pseudo_bci/shaders/arm.frag。ロボット・仮想アイテムの描画にも使う）
// ロボットアームのフラグメントシェーダー（Blinn-Phong の金属っぽい陰影）
// 平面を真上から見ているので、視線は +z 方向から。光は左上・手前から当てる。
in vec3 vNormal;
in vec3 vWorld;

uniform vec3 uColor;
uniform float uAlpha;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(-0.45, 0.55, 0.70));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 Hv = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, Hv), 0.0), 40.0);
    float rim = pow(1.0 - max(N.z, 0.0), 3.0);
    vec3 color = uColor * (0.18 + 0.85 * diff) + vec3(1.0) * spec * 0.55 + uColor * rim * 0.25;
    FragColor = vec4(pow(color, vec3(1.0 / 1.15)), uAlpha);
}
