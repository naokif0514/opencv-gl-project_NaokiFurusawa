# BCI Operator（BCI オペレーター）

BCI でロボットを操作するオペレーターになるゲームです（02_pseudo_bci をもとに、ゲーム性を加えたもの）。
実際の BCI 研究の「ハイブリッド BCI」をもとに、視線（行き先・対象の選択）と、手の動き・握り（擬似ニューロンの発火 → デコード → ロボットアームとグリッパー）を組み合わせます。
横には本物の脳の 3D モデルを表示し、今使っている脳の領域が光ります。

> 現在は **段階 1（本物の脳モデル）** までの実装です。視線と AR、ロボットの移動（段階 2）、ゲームとしての完成（段階 3）、仕上げ（段階 4）はこれから追加します。

## セットアップと実行

仮想環境は一番上の階層の `.venv` を共用します。03 では nilearn を追加で使います。

```bash
cd opencv-gl-project_NaokiFurusawa
.venv/bin/pip install -r 03_bci_game/requirements.txt

cd 03_bci_game
../.venv/bin/python brain_test.py          # 段階 1 の確認用テスト画面（カメラで手を追跡。なければマウス）
../.venv/bin/python brain_test.py --test   # マウスを手の代わりにする
```

### 初回起動（標準脳データの取得）

- **初回はネットワーク接続が必要です。** nilearn が Destrieux アトラス（約 180 KB）を NITRC からダウンロードし、
  fsaverage5（nilearn に同梱）と合わせて `assets/brain_fsaverage5.npz`（約 700 KB）にまとめて保存します。
  2 回目以降はこのファイルだけを読むので、オフラインで動きます（nilearn も読み込みません）。
- **別のパソコンでオフラインで動かすには**、一度ネットワークのある環境で起動してできた `assets/brain_fsaverage5.npz` を、
  相手の `03_bci_game/assets/` にコピーしてください。このファイルがあれば、nilearn もネットワークも要りません。
  （FreeSurfer のデータを含むので、git には入れていません。下の「データのライセンスと出典」を参照）
- **フォールバック（簡易モデル）**：キャッシュがなく、ネットワークにもつながらない（または nilearn が入っていない）ときは、
  左右の半球を楕円体で近似した簡易モデルで起動し、画面に「標準脳データ未取得：簡易モデルで表示中」と表示します。
  領域（運動野・体性感覚野・前頭眼野・視覚野）は MNI 座標でのおおよその位置（中心溝の位置を直線で近似、後頭葉は y < −82 mm など）
  を式で決め、擬似ニューロンもその上の手の領域のあたりに置くので、ゲームはそのまま遊べます。
- データの出典：FreeSurfer の標準脳 fsaverage（Fischl ら 1999）と Destrieux アトラス（Destrieux ら 2010）。くわしくは最後の節を参照。

## 段階 1：本物の脳モデルのテスト画面（`brain_test.py`）

| キー・操作 | 動作 |
|---|---|
| 1 / 2 / 3 / 4 | 運動野 / 体性感覚野 / 前頭眼野 / 視覚野 を光らせる（押すたびに点灯・消灯） |
| 0 | 手で光らせた領域をすべて消す |
| I | pial（本物のしわ）⇔ inflated（ふくらませた表面）を、変形アニメーション（約 2 秒）で切り替える |
| Z | 左半球の手の領域（擬似ニューロンのあるところ）に寄る / 全体に戻る |
| マウスのドラッグ / ホイール | 脳を回転 / 拡大縮小（3 秒操作しないと、またゆっくり自動で回る） |
| ↑ / ↓ | ニューロン数を増やす / 減らす |
| N / M | ノイズを強く / 弱くする |
| K | ニューロンの 1 割を脱落させる |
| T | テストモード（マウス）とカメラモードの切り替え（テストモードでは右下の hand pad の上でマウスを動かす） |
| D | 手の追跡のデバッグ用ウィンドウの表示切り替え |
| ESC | 終了 |

手（またはマウス）を動かすと、02 と同じように擬似ニューロンが発火して光り（色 = 好きな方向）、
そのニューロンのある運動野（手の領域を含む中心前回）も、手の動きの強さに応じて光ります。

### 光らせる領域（Destrieux アトラスのラベル）

| 領域 | 使ったラベル | 説明 |
|---|---|---|
| 運動野（一次運動野 M1） | `G_precentral`（中心前回） | 体の各部の運動を指令する。手の領域は中心前回が後ろへ Ω 字型に張り出した「hand knob」（Yousry ら 1997） |
| 手の領域（hand knob） | 左半球の `G_precentral` のうち、MNI 座標 (-38, -22, 58) から 16 mm 以内 | 擬似ニューロンを置く場所。右手の運動は主に左半球の M1 が制御する（運動の神経が延髄で左右に交差するため）ので、右手で操作する想定で左半球に置いた |
| 体性感覚野（S1） | `G_postcentral`（中心後回） | 触覚・手の位置の感覚を受け取る |
| 前頭眼野（FEF）のあたり | `G_front_middle`（中前頭回）の後ろ側 30% | 目の動き（どこを見るか）を決める。正確には中心前溝と上前頭溝の交わるあたりにあり、アトラスに専用のラベルがないので近い部分で代用した |
| 視覚野 | `Pole_occipital`、`S_calcarine`、`G_cuneus`、`G_oc-temp_med-Lingual` | 後頭葉。一次視覚野（V1）は鳥距溝のまわりにある |

### pial と inflated の変形

pial（脳の本当の形）と inflated（しわを伸ばしてふくらませた形）は、FreeSurfer で同じ頂点番号が皮質の同じ場所に対応しています。
そこで、頂点データに両方の位置を入れておき、頂点シェーダーで 2 つの位置を割合 `uMorph`（0〜1）で混ぜることで、脳がなめらかにふくらむ変形アニメーションにしています。
擬似ニューロンは「どの三角形の、どの重心座標の位置にあるか」で置いているので、両方の表面での位置が求まり、変形中も表面に乗ったまま動きます（領域のラベルも同様）。
テスト画面とゲームのタイトル画面は pial、ゲームを始めるときに pial → inflated へふくらみ（「脳と接続中」の演出）、ゲーム中は inflated で表示します（inflated は溝の奥の領域も見えるため）。

### 表示の設定（`config.py`）

| 設定 | 内容 |
|---|---|
| `BRAIN_SURFACE` | テスト画面の最初の表面：`"pial"`（本物の脳のしわ。既定）/ `"infl"`（ふくらませた表面。溝の中の領域も見える） |
| `MORPH_SEC` | pial ⇔ inflated の変形アニメーションの時間 [秒] |
| `HAND_KNOB_RADIUS` | 手の領域の大きさ [mm] |
| `NEURON_RADIUS_MM` | ニューロンの球の大きさ [mm] |
| `BRAIN_AUTO_ROTATE` | 自動回転の速さ [度/秒] |
| `MSAA_SAMPLES` | アンチエイリアス（0 にすると軽くなる） |

## ファイル構成（段階 1 の時点）

| ファイル | 役割 | 流用元 |
|---|---|---|
| `brain_model.py` | fsaverage5 と Destrieux アトラスの取得・キャッシュ、光らせる領域、手の領域への点の配置 | 新規 |
| `brain_renderer.py` | 脳の表面と擬似ニューロンの描画、回転するカメラ | 新規（ニューロンの描画は 02 の renderer.py と同じ方法） |
| `brain_test.py` | 段階 1 の確認用テスト画面 | 新規 |
| `jtext.py` | 日本語の文字の描画（Pillow + Noto Sans CJK） | 新規 |
| `hand_tracker.py` | 手の追跡（MediaPipe Hands）、テストモードのマウス入力 | 02_pseudo_bci からコピー |
| `neuron_model.py` | 運動野ニューロン集団（コサインチューニング、ポアソンスパイク、ノイズ、脱落） | 02_pseudo_bci からコピー |
| `decoder.py` | PV 法・線形デコーダ | 02_pseudo_bci からコピー |
| `gl_utils.py` | 行列・シェーダー・アイコスフィア | 02_pseudo_bci からコピー |
| `gfx2d.py` | 2D 図形・テクスチャの描画 | 02_pseudo_bci/renderer.py の一部を取り出した |
| `config.py` | 設定 | 02_pseudo_bci/config.py をもとに追加・変更 |
| `shaders/brain.*` | 脳の表面のシェーダー | 新規 |
| `shaders/neuron.*`、`flat2d.*`、`textured.*` | ニューロン・2D 図形・テクスチャのシェーダー | 02_pseudo_bci からコピー |

## データのライセンスと出典

脳のデータは **このリポジトリには含めず**（`.gitignore` で除外）、初回起動時に nilearn で取得します。

- **fsaverage5**（FreeSurfer の標準脳の皮質表面）
  - FreeSurfer（Laboratory for Computational Neuroimaging, Martinos Center, Massachusetts General Hospital）の配布物の一部で、nilearn に同梱されているものを使用
  - ライセンス：FreeSurfer Software License Agreement（https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense）
  - 出典：Fischl, B., Sereno, M. I., Tootell, R. B. H., & Dale, A. M. (1999). High-resolution intersubject averaging and a coordinate system for the cortical surface. *Human Brain Mapping*, 8(4), 272–284.
- **Destrieux アトラス**（aparc.a2009s。皮質の回・溝のラベル）
  - FreeSurfer に含まれるアトラス。nilearn の `fetch_atlas_surf_destrieux` で NITRC（https://www.nitrc.org/）から取得
  - ライセンス：FreeSurfer の配布物として、上と同じ FreeSurfer のライセンスに従う
  - 出典：Destrieux, C., Fischl, B., Dale, A., & Halgren, E. (2010). Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature. *NeuroImage*, 53(1), 1–15.
- **nilearn**（データの取得に使用）：BSD ライセンス。https://nilearn.github.io/
- 手の領域の位置：Yousry, T. A., et al. (1997). Localization of the motor hand area to a knob on the precentral gyrus. A new landmark. *Brain*, 120(1), 141–157.
