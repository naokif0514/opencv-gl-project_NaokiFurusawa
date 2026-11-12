#version 330 core
// （コピー元：02_pseudo_bci/shaders/neuron.frag）
// ニューロンのフラグメントシェーダー
//   ふだん  : 好きな方向の色で暗めに表示（拡散反射のみ）
//   発火率が高い（手の動きの向きが好きな方向に近い）: その色で強く発光する
//   スパイク: その瞬間だけ少し明るくなる
//   脱落    : 灰色で暗く表示
in vec3 vNormal;
in vec3 vWorld;
in vec3 vColor;
in float vFlash;
in float vAlive;
in float vActivity;

uniform vec3 uCamPos;
uniform vec3 uLightDir;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    float diff = max(dot(N, normalize(uLightDir)), 0.0);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);

    vec3 color;
    if (vAlive < 0.5) {
        color = vec3(0.18) * (0.4 + 0.6 * diff);
    } else {
        vec3 base = vColor * (0.10 + 0.22 * diff);
        vec3 glow = vColor * (vActivity * 2.4 + vFlash * 0.45) + vec3(1.0) * vActivity * vFlash * 0.35;
        color = base + glow + vColor * rim * (0.10 + 0.8 * vActivity);
    }
    // トーンマッピングとガンマ補正
    color = vec3(1.0) - exp(-color * 1.3);
    FragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
