# 見た目の改善の記録（スクリーンショット）

レポート・発表用。すべてテストモード（カメラなし）で撮影したもので、人物は写っていない。
同じ乱数の種で物を配置し、プレイ開始 6 秒の画面を撮った（ソフトウェア描画は fps が違うので、配置が少し違う）。

## 段階 B：ライティングと質感、画質の 3 段階

| ファイル | 内容 |
|---|---|
| `stageB_0_before.png` | 変更前（GPU、描く解像度 1.25 倍、MSAA なし、1 つの光 + 半球の環境光） |
| `stageB_1_light.png` | 画質 light（ソフトウェア描画。変更前と同じシェーダー・解像度。玉の色だけ黄 → 赤） |
| `stageB_2_standard.png` | 画質 standard（1.5 倍、MSAA 4、3 つの光 + 環境光、Blinn-Phong + Fresnel、金属の映り込み、木目、フォグ） |
| `stageB_3_high.png` | 画質 high（2.0 倍、MSAA 8、メッシュを細かく） |
| `stageB_4_compare_closeup.png` | ロボットのまわりを拡大して並べたもの（左から 変更前・standard・high） |

fps（カメラモード、上限 60）：変更前 58、light（ソフトウェア描画）28、standard 58、high 58。

## 段階 C：ロボットのモデルの作り込み

ロボットを、`tools/make_robot_model.py`（Python のコード）で組み立てた自作のモデル `assets/models/robot.obj`（約 3,200 三角形）に置きかえた。
拡大の画像は、視点をロボットに近づけて追いかける設定で撮った（ゲームの視点ではない）。

| ファイル | 内容 |
|---|---|
| `stageC_0_before_closeup.png` | 変更前のロボット（拡大、画質 high）：箱 + 球 + 円柱 |
| `stageC_1_after_closeup_high.png` | 変更後（拡大、画質 high）：面取りした車体・デッキ・バッテリー・アンテナ、スポークつきの車輪、発光リングつきの関節、平行グリッパー |
| `stageC_2_after_closeup_light.png` | 変更後（拡大、画質 light = ソフトウェア描画） |
| `stageC_3_after_holding_box.png` | 箱を持っているところ（指は箱の表面で止まる） |
| `stageC_4_after_standard.png` | ゲームの視点（画質 standard） |
| `stageC_5_compare_closeup.png` | 拡大の比較（左 = 変更前、右 = 変更後） |

fps（カメラモード、上限 60）：light（ソフトウェア描画）約 26（段階 B では約 27）、standard 58、high 58。

## 物とカゴの作り込み（段階 D の前）

物とカゴを、`tools/make_item_models.py` で組み立てた自作のモデル `assets/models/items.obj` に置きかえた（画質 standard / high のみ。light は以前の簡単な形のまま）。
大きさ・高さは以前の形と同じなので、視線で選ぶときの投影・吸い付く範囲、つかむ判定は変わらない。
どの画像もボーナスを 1 個置いた状態で撮った。

| ファイル | 内容 |
|---|---|
| `items_0_before_standard.png` | 変更前（画質 standard）：箱 = 立方体、缶 = 円柱、玉 = 球、ボーナス = 立方体 2 つ、カゴ = 板の箱 |
| `items_1_after_standard.png` | 変更後（画質 standard） |
| `items_2_before_closeup.png` | 変更前（拡大、画質 high） |
| `items_3_after_closeup_high.png` | 変更後（拡大、画質 high）：テープ・ラベルつきの段ボール箱、ふち・プルタブ・ラベルの帯つきの缶、縫い目つきのボール、ふくらんだ星 + 回る輪 + 光の粒 + ぼんやりした光、網目とふちのカゴ |
| `items_4_after_light.png` | 変更後の画質 light（以前の簡単な形のまま） |
| `items_5_compare_closeup.png` | 拡大の比較（左 = 変更前、右 = 変更後） |

fps（テストモード、上限なし、変更前と後を交互に 2 回ずつ測った平均）：standard 約 186 → 約 143、high 約 147 → 約 130（上限 60 には十分余裕がある）。
