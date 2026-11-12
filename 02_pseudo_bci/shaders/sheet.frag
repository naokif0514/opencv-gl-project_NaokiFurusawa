#version 330 core
// 皮質の曲面のフラグメントシェーダー：暗いピンク色に、うっすら格子模様をつける
in vec3 vNormal;
in vec3 vWorld;
uniform vec3 uLightDir;
uniform float uActivity;   // 集団全体の活動量 0〜1（多く発火しているほど少し明るく）
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNormal);
    float diff = max(dot(N, normalize(uLightDir)), 0.0);
    vec3 base = vec3(0.30, 0.13, 0.18) * (0.35 + 0.65 * diff);
    // 格子模様（電極アレイのイメージ）
    vec2 g = abs(fract(vWorld.xz * 6.0) - 0.5);
    float line = 1.0 - smoothstep(0.0, 0.04, min(g.x, g.y));
    base += vec3(0.10, 0.05, 0.08) * line;
    base += vec3(0.10, 0.04, 0.10) * uActivity;
    FragColor = vec4(pow(base, vec3(1.0 / 2.2)), 1.0);
}
