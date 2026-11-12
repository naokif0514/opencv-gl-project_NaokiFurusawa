# OpenCV × OpenGL 作品集

OpenCV（カメラ・画像処理）と OpenGL（シェーダーによる 3D 描画）を組み合わせた作品を 3 つ収めています。

| フォルダ | 作品 | 概要 |
|---|---|---|
| [01_focus_brain/](01_focus_brain/) | Focus Brain | Web カメラで顔と目を追跡し（MediaPipe Face Mesh）、まばたき・目の開き・顔の向き・視線の安定性から「集中度」を推定する。集中度に応じて 3D の脳が光る |
| [02_pseudo_bci/](02_pseudo_bci/) | Pseudo BCI Decoder | Web カメラで手を追跡し（MediaPipe Hands）、その動きに応じて運動野ニューロン集団の発火をシミュレーションする。発火パターンだけから手の動きをデコード（PV 法・線形デコーダ）してカーソルとロボットアームを動かす。自動実験モードでデコーダの性質を比べられる |
| [03_bci_game/](03_bci_game/) | BCI Operator（制作中） | 02 をもとにしたゲーム。視線（行き先・対象の選択）と手の動き・握り（擬似ニューロン → デコード → ロボットアーム）を組み合わせた「ハイブリッド BCI」でロボットを操作する。本物の脳の 3D モデル（FreeSurfer fsaverage5 + Destrieux アトラス）の、今使っている領域が光る。現在は段階 1（脳モデルのテスト画面）まで |

くわしい説明・キー操作は、それぞれのフォルダの README.md にあります。

## セットアップ（初回のみ）

すべての作品で、一番上の階層の仮想環境 `.venv` を共用します。

```bash
cd opencv-gl-project_NaokiFurusawa
python3 -m venv .venv
.venv/bin/pip install -r 01_focus_brain/requirements.txt -r 02_pseudo_bci/requirements.txt -r 03_bci_game/requirements.txt
```

`python3 -m venv` が「ensurepip is not available」で失敗する場合は、次のどちらかを行ってください。

- `sudo apt install python3.12-venv` を実行してから、もう一度 `python3 -m venv .venv` を実行する
- sudo が使えない場合は、pip なしで仮想環境を作り、公式の get-pip.py で pip を入れる
  ```bash
  python3 -m venv --without-pip .venv
  python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
  .venv/bin/python /tmp/get-pip.py
  .venv/bin/pip install -r 01_focus_brain/requirements.txt -r 02_pseudo_bci/requirements.txt -r 03_bci_game/requirements.txt
  ```

## 起動

```bash
# 01：集中度で光る 3D の脳
cd 01_focus_brain && ../.venv/bin/python main.py

# 02：擬似 BCI デコーダ体験
cd 02_pseudo_bci && ../.venv/bin/python main.py

# 02 の自動実験モード（画面なし。約 30 秒で results/ に CSV とグラフを保存）
cd 02_pseudo_bci && ../.venv/bin/python main.py --experiment

# 03：BCI オペレーター（制作中。現在は段階 1 の脳モデルのテスト画面）
cd 03_bci_game && ../.venv/bin/python brain_test.py
```

03 は追加で nilearn を使います（`.venv/bin/pip install -r 03_bci_game/requirements.txt`）。
初回だけインターネット接続が必要で、標準脳のデータを `03_bci_game/assets/` に保存します。2 回目以降はオフラインで動きます。

どの作品も、カメラが見つからない場合はテストモード（キーやマウスで操作）で起動します。
ログ（CSV）は、それぞれのフォルダの `logs/` に保存されます。

## 動作環境についての注意

- Ubuntu の Wayland 環境では、glfw（Wayland 版）が異常終了したり、PyOpenGL がコンテキストを見つけられなかったりします。そのため、各 `main.py` の先頭で環境変数を設定し、X11 互換モード（XWayland）でウィンドウを開いています。
- GPU が使えない環境ではソフトウェア描画（llvmpipe）になります。描画が重い場合は、各フォルダの `config.py` にある品質の設定（`MSAA_SAMPLES` など）を下げてください。
