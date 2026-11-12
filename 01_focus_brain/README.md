# Focus Brain（集中度で光る・育つ 3D の脳）

Web カメラでユーザーの「集中度」を推定し、その値に応じて画面中央の 3D の脳が光ったり育ったりする作品です。
OpenCV（カメラ取得・画像処理・頭部姿勢推定）と OpenGL（シェーダーによる 3D 描画）を両方使っています。

> 現在は **最小構成**（カメラ → 集中度の数値表示 → 脳が集中度で光る）までの実装です。
> パーティクル・シナプス・成長の表現は次の段階で追加します。

## セットアップ（Ubuntu 24.04 / Python 3.10 以上）

```bash
cd opencv-gl-project_NaokiFurusawa

# 仮想環境を作ってライブラリを入れる
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`python3 -m venv` が「ensurepip is not available」で失敗する場合は、次のどちらかを行ってください。

- `sudo apt install python3.12-venv` を実行してから、もう一度 `python3 -m venv .venv` を実行する
- sudo が使えない場合は、pip なしで仮想環境を作り、公式の get-pip.py で pip を入れる
  ```bash
  python3 -m venv --without-pip .venv
  python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
  .venv/bin/python /tmp/get-pip.py
  .venv/bin/pip install -r requirements.txt
  ```

## 実行

```bash
.venv/bin/python main.py              # カメラを使って起動
.venv/bin/python main.py --no-camera  # カメラなし（テストモード）で起動
.venv/bin/python main.py --camera 1   # 0 番以外のカメラを使う
```

カメラが見つからない場合は、自動的にテストモードで起動します。

## キー操作（3D ウィンドウで）

| キー | 動作 |
|---|---|
| ESC | 終了して、集中度の推移を `logs/focus_日付_時刻.csv` に保存 |
| D | デバッグ用カメラウィンドウの表示切り替え |
| R | 成長状態をリセット（成長の表現は次の段階で追加） |
| 1 / 2 | 押している間、集中度を下げる / 上げる（テストモード） |
| 0 | テストモードをやめて、カメラによる推定に戻る |
| C | 今の顔の向きを「正面」として登録する |

カメラが画面の真正面にない場合（ノート PC の外付けカメラが横や下にある場合など）は、画面を見た状態で **C** を押してください。押さないと、画面を見ていても「よそ見」と判定されて集中度が低く出ます。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `main.py` | 起動、キー操作、メインループ、デバッグウィンドウ、終了時のログ保存 |
| `camera_tracker.py` | カメラ処理スレッド。MediaPipe Face Mesh で特徴点を検出し、各指標を計算 |
| `focus_estimator.py` | 各指標を 0〜1 のスコアに正規化し、重み付き合計と EMA で集中度を算出 |
| `renderer.py` | glfw + PyOpenGL による描画（脳、HUD） |
| `brain_mesh.py` | 脳メッシュの生成（プロシージャル）と OBJ 読み込み |
| `session_logger.py` | 集中度の推移を CSV に保存 |
| `config.py` | しきい値・重み・表示パラメータ |
| `shaders/` | GLSL シェーダー（脳、HUD） |
| `assets/brain.obj` | （任意）置くとプロシージャルな脳の代わりに読み込む |

## 動作環境についての注意

- Ubuntu の Wayland 環境では、glfw（Wayland 版）が異常終了したり、PyOpenGL がコンテキストを見つけられなかったりします。そのため、`main.py` の先頭で X11 互換モード（XWayland）を使うよう環境変数を設定しています。
- GPU が使えない環境（`[renderer] OpenGL ...` の行のあとの描画が重い場合）では、ソフトウェア描画（llvmpipe）になります。`config.py` の `BRAIN_DETAIL` や `MSAA_SAMPLES` を下げると速くなります。
