# EchoLoop v5 — シミュレーション結果

## 仮説

> ループが完全分離でなく部分的に route を共有することで、
> mixed internal state / blended attractors / partial synchronization が生まれる。

---

## v4 からの設計変更

| 項目 | v4 | v5 |
|---|---|---|
| ループ構造 | 完全分離 (9 routes, 3 per loop) | **shared bridge routes あり** (11 routes) |
| ループサイズ | 全て 3-route | **社会/防御=4-cycle, 探索=5-cycle** |
| shared routes | なし | **observe (社会↔探索), vigilance (探索↔防御)** |
| cross-inhibition | 全ルートに等しく | **shared route は非メンバーからのみ (50% 減)** |
| 疲労伝播 | ループ内のみ | **edge の所属 loop で決定 → bridge 経由で伝播** |
| switching 回数 | 4 | **21** |
| 平均 ambiguity | — | **0.598** (v3/v4 は計測なし) |

---

## アーキテクチャ

```
social (4-cycle):      attention → observe → engage → approach → (→attention)
                                     ↑↑↑  bridge  ↓↓↓
exploration (5-cycle): curiosity → observe → wander → vigilance → inspect → (→curiosity)
                                                           ↑↑↑  bridge  ↓↓↓
defensive (4-cycle):   alert     → vigilance → freeze → withdraw → (→alert)
```

### shared route の役割

| route | 所属 | 機能 |
|---|---|---|
| `observe` | social(0.70) + exploration(0.70) | 社会注意と探索注意が同一 node を争う |
| `vigilance` | exploration(0.60) + defensive(0.60) | 探索の警戒と防御警戒が同一 node を争う |

### cross-inhibition の非対称性

| route の種類 | 抑制元 | 相対強度 |
|---|---|---|
| exclusive (attention など) | 2 non-member loops から | 1.0x |
| shared (observe, vigilance) | 1 non-member loop から | **0.5x** |

shared route は抑制が少ない → ループが切り替わっても bridge が消えにくい → blending が持続

---

## 実験フェーズ (N=700 steps)

| フェーズ | ステップ | 外界入力 |
|---|---|---|
| internal-1 | 0–100 | なし (social seed) |
| soc push | 100–160 | attention+0.20, engage+0.15 |
| internal-2 | 160–270 | なし |
| exp push | 270–340 | curiosity+0.20, wander+0.15 |
| internal-3 | 340–470 | なし |
| def push | 470–540 | alert+0.20, freeze+0.15 |
| internal-4 | 540–700 | なし |

---

## 実行結果 (seed=42, 700 steps)

### Switching と Dominance

| ループ | ステップ数 | 割合 | 平均 dwell |
|---|---|---|---|
| `social` | 245 | 35% | 61.2 steps |
| `exploration` | 95 | 14% | **9.5 steps** |
| `defensive` | 360 | 51% | 45.0 steps |
| **total switches** | **21** | — | — |

### Blended State 指標

| 指標 | 値 | 解釈 |
|---|---|---|
| Mean dominance ambiguity | **0.598** | 0=完全支配, 1=完全同等。平均6割が曖昧 |
| Mean blend coefficient | **0.344** | 平均34%が非支配ループから |

### Bridge Route 活性化

| route | mean act | max act |
|---|---|---|
| `observe` | 0.086 | 0.732 |
| `vigilance` | 0.096 | 0.918 |

bridge route は平均は低いが、スパイク時に 0.7-0.9 まで達する。共有争奪が激しい時間帯がある。

---

## 観察・考察

### 1. Switching 頻度の劇増 (4 → 21 回)

v4 (完全分離ループ) では 4 回だったのが、v5 では 21 回に増加。  
shared route がループ間の「漏れ」を作り、支配が不安定化した。  
特に `exploration` の mean dwell = **9.5 steps** は非常に短く、  
両側から (`observe` で social, `vigilance` で defensive) 常に引き合われている。

### 2. Mean ambiguity = 0.598

全ステップの平均で 60% が ambiguous (= 2 位ループとの差が小さい) 状態。  
これは「どのループが dominant か曖昧」な時間が大半を占めることを意味する。  
v3/v4 の明確な winner-take-all ダイナミクスとは質的に異なる。

### 3. Idle 70% = 行動的曖昧さ

idle が 70% と高い。これは blending による activation の分散が原因。  
複数ループが中程度に活性化すると、どの route も threshold を超えない。  
→「どう行動すべきか曖昧なとき、行動が出にくい」という mood-like 現象。

### 4. Bridge route の spike pattern

`vigilance` は最大 0.918 に達するスパイクがある。  
これは exploration と defensive が同時に vigilance を高めた瞬間。  
一方 `observe` は最大 0.732 — social と exploration が同調した瞬間に amplified される。  
→ 「部分的同期によるスパイク増幅」が bridge node で観測される。

### 5. 非対称な dwell 分布

- social: 安定 (61 step mean) — exclusive routes が多く外乱を受けにくい
- exploration: 不安定 (9.5 step mean) — 両側 bridge に引き合われ支配できない
- defensive: 中間 (45 step mean) — vigilance が partial bridge だが alert/freeze/withdraw が exclusive で安定

これは「橋渡し役のループが最も不安定」という構造から emergent に生まれた。

### 6. Rolling Correlation (予測)

`observe` 共有: social↔exploration の相関は共有活性化時に正、競合時に負  
`vigilance` 共有: exploration↔defensive の相関は同様のパターン  
social↔defensive (bridge なし): 全期間を通じて弱い相関  
→ 共有 route の有無が同期パターンを決定する

---

## 可視化 (`echoloop_v5_result.png`)

| パネル | 内容 |
|---|---|
| Loop Activity (上段全幅) | 実線=ループ活性度、破線=shared routes (observe/vigilance)。背景白=ambiguity高 |
| Ternary Diagram (中段) | 3ループを三角形の頂点に配置した状態軌道。中心 = 完全 blending |
| Ambiguity + Blend (中段) | 支配曖昧さと blend 係数の時系列 |
| Bridge vs Exclusive (中段) | observe/vigilance vs 排他的 route の比較 |
| Rolling Synchronization (中段) | ループ間 Pearson r の推移。bridge 有/無で差が出るか |
| Phase Spaces (下段×2) | social×exploration, exploration×defensive。bubble=ambiguity |
| Route Activations (下段) | 全 11 route (bridge route を太線で強調) |
| Action Distribution (下段) | idle 70% = behavioral ambiguity の可視化 |

---

## v1〜v5 の設計進化まとめ

| v | 核心 | スイッチ数 | 平均曖昧さ |
|---|---|---|---|
| 1 | route graph + Hebbian | — | — |
| 2 | 単一 limit cycle | 0 | 0 |
| 3 | 3-loop 競合 (完全分離) | 2 | 低 |
| 4 | 疲労 + 完全分離 | 4 | 低 |
| **5** | **shared bridge routes** | **21** | **0.598** |

---

## 次の実験案

- **bridge 強度の調整**: observe の membership weight を変えると ambiguity がどう変わるか
- **3つ目の bridge**: 社会↔防御を繋ぐ route を追加した場合、3ループ同時 blending が起きるか
- **speak() の実装**: `observe` スパイク時 + social 支配時に発話を読み出す
- **同期指数の制御**: bridge weight で意図的に同期/非同期を制御できるか
- **bridge route への fatigue**: observe/vigilance 自体が疲弊したら何が起きるか
