# EchoLoop v4 — シミュレーション結果

## 仮説

> 状態履歴そのものが dynamics を変える。  
> dominant loop は疲労し、次第に自律維持できなくなる。  
> 非支配 loop はゆっくり回復し、次の dominance 候補になる。  
> 「mood-like dynamics」が内部状態の振る舞いとして emergent に出る。

---

## v3 からの設計変更

| 項目 | v3 | v4 |
|---|---|---|
| attractor | 静的構造 | **使用履歴によって変化** |
| ループ安定性 | 疲労なし (常に固定固有値) | **fatigue が effective gain を動的に変化** |
| 切り替えトリガ | 外界入力のみ | **疲労 + 外界入力 + ノイズ** |
| spontaneous switching | なし (噂入力なしでは切り替わらない) | **あり (内部疲労で自発的に崩壊)** |

---

## 疲労メカニズム

```python
# 毎 step:
fatigue[l] += FATIGUE_RATE * loop_activity[l]  # 活性度に比例して蓄積
fatigue[l] *= FATIGUE_DECAY                     # 全ループが緩やかに回復

# forward edge への適用:
eff_w = W_FWD * max(MIN_GAIN, 1.0 - fatigue[l])

# 回転固有値への影響:
|λ_rot(f)| = sqrt(DECAY^2 + (eff_w + W_BWD)^2)
```

### 疲労と固有値の関係

| fatigue | eff_W_fwd | \|λ_rot\| | 状態 |
|---|---|---|---|
| 0.0 | 0.550 | 1.238 | 健全・自律維持 |
| 0.4 | 0.330 | 1.089 | やや疲労 |
| 0.6 | 0.220 | 1.023 | かなり疲労 |
| **0.8** | 0.110 | **0.967** | **崩壊境界 (threshold)** |
| 1.0 | 0.028 | 0.930 | 自律維持不能 |

疲労 0.80 を超えると `|λ_rot| < 1` → ループが自律的に崩壊し、支配が失われる。

---

## 実験フェーズ (N=800 steps)

| フェーズ | ステップ | 外界入力 | 意図 |
|---|---|---|---|
| internal-1 | 0–120 | なし | social 起動 |
| exp trigger | 120–160 | exploration routes += 0.22 | exploration への push |
| internal-2 | 160–450 | なし | **長い pure-internal wandering** |
| def trigger | 450–490 | defensive routes += 0.22 | defensive への push |
| internal-3 | 490–700 | なし | stuck state / recovery 観測 |
| mixed | 700–800 | 弱い混合入力 | metastable 挙動 |

---

## 実行結果 (seed=42, 800 steps)

### 切り替え履歴

| 期間 | 支配ループ | 長さ | 切り替え原因 |
|---|---|---|---|
| 0–119 | **social** | 120 steps | 初期 seed |
| 120–204 | **exploration** | 85 steps | exp trigger (120-160) で seeding |
| 205–404 | **social** | 199 steps | exploration の疲労 → social が回復して再支配 |
| 405–495 | **exploration** | 91 steps | social が 199 step の長期疲労で崩壊 |
| 496–800 | **defensive** | 305 steps | def trigger (450-490) で seeding → **stuck state** |

### ループ支配分布

| ループ | ステップ数 | 割合 |
|---|---|---|
| `social` | 319 | 40% |
| `exploration` | 176 | 22% |
| `defensive` | 305 | **38%** |

### ドウェル時間

| ループ | 回数 | mean | min | max |
|---|---|---|---|---|
| social | 2 | **159.5** steps | 120 | 199 |
| exploration | 2 | 88.0 steps | 85 | 91 |
| defensive | 1 | **305.0** steps | — | — |

### 最終疲労状態

| ループ | 疲労 | 解釈 |
|---|---|---|
| social | 0.213 | 疲弊から回復中 |
| exploration | 0.404 | 中程度疲労 (前の 91-step 支配の残り) |
| defensive | 0.154 | まだ支配中 (蓄積 < 0.80) |

---

## 観察・考察

### 1. Spontaneous Switching (確認済み)

**step 205**: exploration loop が自発的に崩壊し social が再浮上。  
external input は step 160 で終了しており、205 のスイッチは exploration の疲労蓄積による内発的崩壊。  
→「外界入力なしで spontaneous switching が起きた」仮説を確認。

**step 405**: social loop が 199 step の長期支配後に疲労崩壊。  
これも pure-internal (internal-2 フェーズ, 外界入力なし) での spontaneous switch。

### 2. Stuck State / Hysteresis (確認済み)

defensive loop は step 496 から 800 まで **305 step 連続支配**。  
この間、social と exploration の疲労が蓄積されており回復に時間がかかった。  
mixed フェーズ (700-800) で弱い social/exploration 入力があっても切り替わらなかった。  
→「特定ループが stuck になる現象」が発生。

### 3. 不対称なドウェル時間

social の mean dwell (159.5 steps) >> exploration の mean dwell (88.0 steps)。  
social は初期 seed で高い活性度から始まるため疲労に耐えやすく長く続く。  
exploration は毎回 seeding から始まるため疲労閾値到達が早い。  
→ループの「初期条件」と「疲労耐性」が dwell time の非対称性を生む。

### 4. Low-energy Transitional States (idle 65%)

アクション分布: `idle 65%` は遷移期や疲労期に routes が activation threshold (0.38) 以下になる時間。  
v3 では `idle < 1%` だったのに対し、v4 では疲労により routes のピーク値が下がる → threshold 未達が頻繁に発生。  
これは「疲れた状態 = 行動が少なくなる」という mood-like dynamics の一部。

### 5. Effective Gain の動的変化

dominant loop の effective rotating gain (`|λ| - 1`) は徐々に低下し、  
0 を下回ると自律維持が終了 → spontaneous switch が発生する。  
このとき competing loops の gain は回復中で相対的に高い → 新しい dominant が決まる。  
→「疲れた attractor から元気な attractor へ」のパターンが内部状態から emergent に出た。

---

## 可視化 (`echoloop_v4_result.png`)

| パネル | 内容 |
|---|---|
| Loop Activity + Fatigue (上段全幅) | 実線=ループ活性度、破線=疲労レベル (右軸)、点線=疲労閾値0.80 |
| Effective Gain (中段) | |λ_rot|−1 の推移。0 を下回ると崩壊中。switch タイミングと対応 |
| Dominance Raster (中段) | 各 step の支配ループ + 切り替え点を白線でマーク |
| Dwell Distribution (中段) | 各ループの支配継続時間のヒストグラム |
| Entropy (中段) | 不確かさ推移。switch 直後にスパイク、stuck state で低下 |
| Phase Space social×exploration (下段) | bubble サイズ = 支配ループの疲労量。★=理論的 attractor |
| Phase Space exploration×defensive (下段) | 同上 |
| Route Activations (下段) | 全 9 ルートの推移 |
| Action Distribution (下段) | idle が多い = 疲労による低活性期が多い |

---

## v1〜v4 の設計進化まとめ

| バージョン | 核心 | 主な観察 |
|---|---|---|
| v1 | route graph + Hebbian 強化 | attention が reward で dominance を獲得 |
| v2 | 単一閉ループ + 回転波 | limit cycle の自律持続、blackout でも 100% 持続 |
| v3 | 3 ループ競合 + 相互抑制 | attractor switching、hysteresis |
| **v4** | **疲労/適応による動的 landscape** | **spontaneous switching、stuck state、mood-like oscillation** |

---

## 次の実験案

- **疲労非対称化**: social は recover が速く (社会性は戻りやすい), defensive は recover が遅い
- **ループ長変更**: 2 ノードループ (faster cycling) vs 5 ノードループ (slower)
- **speak() の追加**: social ループ dominant + fatigue < 0.4 のときに発話を読み出す
- **疲労伝染**: dominant loop の疲労が共有 route を通じて隣接ループに伝播する
- **外界刺激の "intensité"**: 強すぎる入力が疲労回復を妨げる仕組み
