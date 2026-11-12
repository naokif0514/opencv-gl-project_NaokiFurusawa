#version 330 core
// 大脳皮質の表面のフラグメントシェーダー
//   ・地の色：溝の奥ほど暗く、回の上ほど明るい（脳の画像でよく使われる表示）
//   ・陰影：拡散反射 + 弱い鏡面反射 + 半球状の環境光（裏から見ても暗くなりすぎないよう両面で照らす）
//   ・領域の発光：uGlow[領域番号]（0〜1）に応じて、その領域を色つきで光らせる。
//     uHint で、光っていないときも領域の位置をうっすら色で示す
in vec3 vNormal;
in vec3 vWorld;
in float vSulc;
flat in int vRegion;

uniform vec3 uCamPos;
uniform vec3 uLightDir;
uniform float uGlow[6];
uniform vec3 uRegionColor[6];
uniform float uHint;
uniform float uTime;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 H = normalize(L + V);

    float gyrus = 1.0 - smoothstep(-0.9, 0.9, vSulc);              // 1 = 回の上、0 = 溝の奥
    vec3 base = mix(vec3(0.42, 0.37, 0.37), vec3(0.88, 0.82, 0.79), gyrus);

    int r = vRegion;
    float glow = uGlow[r];
    if (r == 2) glow = max(glow, uGlow[1]);                         // 手の領域は運動野の一部
    vec3 rc = uRegionColor[r];
    if (r > 0) base = mix(base, rc, uHint);

    float diff = max(dot(N, L), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    float spec = pow(max(dot(N, H), 0.0), 24.0) * 0.18;
    vec3 color = base * (0.28 + 0.22 * hemi + 0.62 * diff) + vec3(spec);

    color = vec3(1.0) - exp(-color * 1.6);                            // トーンマッピング（地の色だけ）
    color = pow(color, vec3(1.0 / 1.4));

    // 光らせる領域：トーンマッピングのあとで色を混ぜる（白っぽくならず、鮮やかな色で光る）
    if (r > 0 && glow > 0.0) {
        float pulse = 0.85 + 0.15 * sin(uTime * 6.0);
        vec3 lit = rc * (0.45 + 0.65 * diff) * (0.9 + 0.25 * pulse);
        color = mix(color, lit, 0.85 * glow) + rc * 0.18 * glow * pulse;
    }
    FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
