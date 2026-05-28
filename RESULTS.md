# EchoLoop — シミュレーション結果

## 仮説

> ニューラルネットではなく「経路」が本体。  
> 外界入力は内部循環ルートを歪めるだけ。行動は内部状態の読み出しとして発火する。

---

## モデル構造

### 世界 (2D 空間)

| エンティティ | 説明 |
|---|---|
| `User` | 座標 + 視線方向 (gaze) + reaction (−1〜+1) |
| `Agent` | 座標 + 内部ルートグラフ |
| `WorldObject` | 座標 + interest (0.6〜1.4) の静的オブジェクト × 4 |

### 内部ルート (5 種)

| ルート | 発火アクション | 主な入力源 |
|---|---|---|
| `attention` | `look_at_user` | ユーザーの視線がエージェントに向いている |
| `curiosity` | `point` | 近くのオブジェクトの interest 総和 |
| `approach` | `move_toward` | ユーザーが近くにいて reaction が正 |
| `observe` | `look_at_object` | attention / curiosity から流入 |
| `idle` | `idle` | 常時ベースライン (+0.04/step) |

各ルートは `activation` と `strength` を持つ。  
`strength` はユーザーの reaction によって強化・減衰される (Hebbian 学習)。

### ノード/エッジグラフ

```
class Node:
    x, y       # 2D 位置
    tag        # 所属ルート名
    activation # 活性度

class Edge:
    a, b       # ノードペア
    strength   # 経路強度 (よく通るほど強くなる)
    usage      # 累積通過量
```

- **80%**: よく使われる edge 周辺に生成  
- **20%**: 完全ランダム生成

ノードグラフはルート動力学への逆フィードバックは持たない。  
構造的学習 (どの経路が強化されたか) の記録・可視化レイヤとして機能する。

### 毎 step の処理順

```
1. 外界入力を計算
      attention_in  = gaze_align × exp(−dist/5)
      curiosity_in  = Σ(interest × exp(−dist/4)) / n_objects
      approach_in   = clip(1 − dist/8) × (0.5 + reaction × 0.5)

2. 外界入力 → ルート activation を押し上げ (distortion)

3. ルート間相互作用 (内部循環)
      例: attention → observe (+0.40)、approach → idle (−0.30)

4. ルート activation → タグ付きノードへ注入

5. ノードグラフ内フロー (edge.strength で重み付け)

6. 減衰 (activation × 0.84 / step)

7. ノード生成 (確率 6%)

8. アクション発火
      score = activation × strength
      score が ACTION_THRESH (0.22) 超えの最高スコアルートが勝つ

9. user.reaction → 勝ちルートの strength を ±0.02 更新

10. edge 強化 / 減衰 (Hebbian)
```

---

## シミュレーション設定

| パラメータ | 値 |
|---|---|
| ステップ数 | 300 |
| ワールドサイズ | 10 × 10 |
| アクション閾値 | 0.22 |
| 視線サイクル長 | 20 steps |

**ユーザー視線フェーズ (20 step 周期)**

| フェーズ | 割合 | 内容 | reaction |
|---|---|---|---|
| gaze_agent | 30% | agent を見る | +0.5 |
| gaze_object | 30% | 最近オブジェクトを見る | 0.0 |
| neutral | 20% | 視線をそらす | −0.15 |
| close_positive | 20% | agent を見て肯定的 | +0.4 |

---

## 実行結果 (seed=42, 300 steps)

### ルート最終状態

| ルート | activation | strength | 変化 |
|---|---|---|---|
| `attention` | 0.342 | **2.169** | ↑ 強化 (ユーザーに頻繁に注目され続けた) |
| `curiosity` | 0.304 | 0.985 | → 横ばい |
| `approach` | 0.266 | 0.996 | → 横ばい |
| `observe` | 0.106 | 1.000 | → 横ばい |
| `idle` | 0.151 | 1.050 | → 微増 |

### アクション分布

| アクション | 回数 | 割合 |
|---|---|---|
| `look_at_user` | 233 | **78%** |
| `point` | 38 | 13% |
| `idle` | 22 | 7% |
| `move_toward` | 7 | 2% |

### ノードグラフ

| 項目 | 初期 | 最終 |
|---|---|---|
| ノード数 | 20 | **44** (24 個生成) |
| エッジ数 | — | 326 |

---

## 観察・考察

### 1. attention ルートの支配化

`attention` の strength が **1.0 → 2.17** に成長した。  
ユーザーが agent を見るたびに reaction=+0.5 が適用され、  
「attention が勝つ → strength が上がる → より勝ちやすくなる」という正のフィードバックループが成立している。  
これは「よく通る経路が強化される」という仮説の動作そのもの。

### 2. 外界入力による distortion

ユーザーが視線をそらす neutral フェーズでは attention_in が落ち、  
curiosity (常時バックグラウンド入力) が相対的に浮上して `point` が発火する (13%)。  
外界入力がルートの競合バランスを変えるという仮説の動作が確認できる。

### 3. ノード生成のバイアス

80% exploit / 20% random の混合戦略により、  
attention/curiosity タグのノードが多く生成され、  
それらのエッジが強化されて密なクラスタを形成した。

### 4. observe が出にくい理由

`observe` は直接の外部入力を持たず、attention/curiosity からの流入に依存する。  
さらに observe のアクション (`look_at_object`) はアクション閾値を超えるほど強くならなかった。  
→ 物体が近くにあるときの observe 増強、または observe の直接外部入力追加が改善案。

---

## 可視化

`echoloop_result.png` に以下の 4 パネルを出力:

| パネル | 内容 |
|---|---|
| World (左) | 2D 軌跡 + アクション種別の色付きドット |
| Route Activations (右上) | 300 step のルート activation 推移、閾値ライン、ユーザー視線フェーズのシェード |
| Internal Route Graph (右下左) | ノード/エッジの構造 (edge.strength をアルファ値で表現) |
| Route Strengths (右下右) | 最終 strength のバーチャート |

---

## 次の実験案

- **observe 強化**: オブジェクトが視野内にあるとき observe への直接入力を加える
- **strength 非対称減衰**: 負の reaction による減衰を強めて「避けるべき行動」を明確化
- **複数エージェント**: ルート dynamics を持つ agent 同士の相互作用
- **ノード可視化のアニメーション**: step ごとのノード activation の変化を動画で確認
- **記憶なし vs 記憶あり比較**: strength を固定した場合との行動分布の差
