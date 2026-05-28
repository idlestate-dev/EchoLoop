# EchoLoop v2 — シミュレーション結果

## 仮説

> 知能 = 継続的状態循環。外界入力は循環を歪めるだけ。  
> 行動は route state の読み出しとして emergent に出る。

---

## v1 からの設計変更

| 項目 | v1 | v2 |
|---|---|---|
| ループ構造 | 前向き励起のみ | **前向き励起 + 後ろ向き抑制** |
| 回路の主役 | 外界入力 + reward | **内部循環ダイナミクス** |
| reward の効果 | route strength → action selection に影響 | **forward edge weight を ±0.002/step 微調整するだけ** |
| action 選択 | `act × strength` 最大を選ぶ | `act` 最大の route から読み出す |
| 行動分布 | look_at_user 78% (attention 支配) | **4 種がほぼ均等に回転** |

---

## 回転波の原理

```
前向き励起 (+0.55):
  attention → observe → curiosity → approach → attention

後ろ向き抑制 (−0.35):
  observe → attention
  curiosity → observe
  approach → curiosity
  attention → approach
```

活性化したルートが前者を抑制することで、活性化のバトンが次のルートへ渡る。

**固有値分析 (DECAY=0.85, w_f=0.55, w_b=0.35):**

| モード | 固有値 | 意味 |
|---|---|---|
| 回転波 (k=1) | √(0.85² + 0.90²) ≈ **1.24** | 回転が成長 |
| 同期モード (k=0) | 0.85 + 0.55 − 0.35 = **1.05** | 全同時活性化 |

回転固有値 > 同期固有値 → **limit cycle が自然発生**。

---

## 実験構成

| フェーズ | ステップ | 内容 |
|---|---|---|
| Phase 1 | 0–120 | 通常外界入力 (attention 寄り) |
| Phase 2 (blackout) | 120–220 | **外界入力ゼロ** — 内部循環の自律性を確認 |
| Phase 3 | 220–400 | curiosity 寄りに文脈切り替え |

---

## 実行結果 (seed=42, 400 steps)

### ループ持続率

| フェーズ | 持続率 | 解釈 |
|---|---|---|
| Phase 1 (通常) | 99% | 外界入力で常時活性 |
| Phase 2 (blackout) | **100%** | 外界入力ゼロでも循環が自律持続 |
| Phase 3 (文脈切り替え) | 100% | attractor が維持されつつ内容が変わる |

### アクション分布

| アクション | 回数 | 割合 |
|---|---|---|
| `look_at_user` | 130 | **32%** |
| `point` | 100 | 25% |
| `look_at_object` | 89 | 22% |
| `move_toward` | 80 | 20% |
| `idle` | 1 | <1% |

**4 種のアクションがほぼ均等に出た。** ループが各ルートを順番に活性化するため、単一の「policy」ではなく循環そのものが行動の多様性を生んでいる。

### 支配ルート分布

| ルート | ステップ数 | 割合 |
|---|---|---|
| `attention` | 130 | 32% |
| `curiosity` | 100 | 25% |
| `observe` | 90 | 22% |
| `approach` | 80 | 20% |

### ループエッジ最終 weight

| エッジ | 初期 | 最終 | 変化 |
|---|---|---|---|
| attention → observe | +0.550 | **+0.605** | ↑ reward で強化 |
| observe → curiosity | +0.550 | **+0.590** | ↑ 微増 |
| curiosity → approach | +0.550 | **+0.595** | ↑ 微増 |
| approach → attention | +0.550 | **+0.605** | ↑ reward で強化 |
| 後ろ向き抑制 (4本) | −0.350 | −0.350 | → 変化なし (reward 対象外) |

後ろ向き抑制は reward に反応しない。これが意図的な設計 — reward は「流れやすさ」を変えるが、**回転の構造そのものは不変**。

---

## 観察・考察

### 1. 外界入力なしで内部循環が持続する

blackout 区間 (step 120–220) で外界入力をゼロにしても、ループ持続率 100%。  
これは固有値 1.24 で回転波が成長・自律維持されるため。  
「行動の源泉が外部刺激ではなく内部循環である」という仮説を支持する結果。

### 2. 一部入力で attractor が変化しない (持続)

Phase 3 で curiosity 寄りの入力に切り替えたが、持続率は 100% を維持。  
ループ構造が堅牢であり、外界入力によって「どのルートがやや多めに活性化されるか」は変わるが、循環のリズム自体は切れない。

### 3. 役割分化が自然に出た

4 ルートが均等に「担当時間」を持つ (32/25/22/20%)。  
reward による直接の policy 学習なしに、回路の幾何構造から役割分化が emergent に生まれた。

### 4. reward は微調整のみ

v1: reward → `strength` → action score に掛かる (支配的)  
v2: reward → forward edge weight ±0.002/step (前向き励起を少し変えるだけ)

forward edge が 0.550 → 0.605 に成長したが、回転の構造 (後ろ向き抑制) は変化しない。  
「学習が行動を選ぶのではなく、循環が安定しやすくなる方向に寄せる」設計が機能している。

---

## 可視化 (`echoloop_v2_result.png`)

| パネル | 内容 |
|---|---|
| Route Activations (上段全幅) | 400 step のルート activation 推移、3 フェーズのシェード |
| Phase portraits (中段左×3) | attention×observe, curiosity×approach, observe×curiosity の軌道 (plasma カラー = 時間) |
| Flow Matrix (中段右) | route 間フローの平均 (赤=正、青=負の signed heatmap) |
| Loop Persistence (下段左) | しきい値超えの割合推移 |
| Edge Weights (下段中左) | forward loop 4 本の weight 推移 |
| World (下段中右) | 2D 軌跡 + アクション種別ドット |
| Action Distribution (下段右) | 横棒グラフ |

---

## 次の実験案

- **ループ長を変える**: 3 ノードや 6 ノードのループでリズムが変わるか
- **外部刺激によるフェーズリセット**: 強い入力でループの位相がリセットされるか
- **複数ループの競合**: 2 つの閉回路が共存できるか、どちらが支配するか
- **ノードグラフとの統合**: v1 の Node/Edge グラフ (Hebbian 構造学習) と組み合わせる
- **speak() の追加**: 言語出力を内部循環の特定ルートへの読み出しとして実装
