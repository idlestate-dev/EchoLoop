# EchoLoop v6 — パラメータ感度スイープ結果

一変数一変動（OAT）スイープ。各設定 20 シード × 3条件。
他パラメータはすべてベースライン値に固定。

**列の説明**
- NC/CNS/FULL = 平均±標準偏差（侵入回数）
- FULL−NC % = FULLの侵入回数がNO_CALLSより低下した割合（正 = FULL優位）
- ✓win = FULLがNO_CALLSを下回ったシードの割合
- CNS% = CNSのNO_CALLS比改善率　△>+5%  ▽<−5%  — = ±5%以内

## n_agents

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| 4 | 5.0±4.2 | 4.4±3.9 | 4.0±3.1 | ✓+18% | 50% | 9.2±7.8 | 7.3±6.4 | +20% |
| 8 | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 15.1±9.5 | +23% |
| 16 | 24.8±10.4 | 21.9±8.8 | 15.8±5.2 | ✓+36% | 95% | 49.4±21.2 | 36.9±16.0 | +25% |

## n_steps

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| 500 | 3.2±2.9 | 3.0±2.3 | 2.5±2.1 | ✓+20% | 30% | 7.0±6.5 | 5.9±5.1 | +16% |
| 1k | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 15.1±9.5 | +23% |
| 2k | 17.4±11.2 | 17.2±11.0 | 13.1±8.1 | ✓+25% | 75% | 34.0±22.1 | 27.5±20.2 | +19% |

## danger_radius

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| r=6 | 7.8±3.9 | 7.2±3.3 | 6.0±2.3 | ✓+24% | 60% | 13.6±7.3 | 11.1±5.4 | +19% |
| r=10 | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 15.1±9.5 | +23% |
| r=14 | 7.3±5.9 | 6.7±5.0 | 5.7±4.1 | ✓+22% | 60% | 15.4±13.2 | 12.6±9.4 | +18% |

## hear_sigma

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| σ=8 | 10.0±7.0 | 9.6±6.1 | 9.3±6.7 | ✓+7% | 55% | 19.6±13.6 | 20.1±13.6 | -2% |
| σ=16 | 10.0±7.0 | 9.1±6.1 | 7.9±5.4 | ✓+21% | 70% | 19.6±13.6 | 16.1±9.8 | +18% |
| σ=24 | 10.0±7.0 | 8.2±5.1 | 7.6±5.0 | ✓+24% | 60% | 19.6±13.6 | 16.5±10.1 | +16% |

## mark_strength

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| 0.5× | 10.0±7.0 | 9.1±5.6 | 8.0±5.4 | ✓+20% | 65% | 19.6±13.6 | 17.6±12.2 | +10% |
| 1× | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 15.1±9.5 | +23% |
| 2× | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 14.9±10.3 | +24% |

## avoid_decay

| 値 | NC entries | CNS entries | FULL entries | FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 10.0±7.0 | 9.1±5.6 | 8.1±5.5 | ✓+19% | 65% | 19.6±13.6 | 15.1±9.4 | +23% |
| 0.95 | 10.0±7.0 | 9.1±5.6 | 7.0±4.4 | ✓+30% | 75% | 19.6±13.6 | 15.1±9.5 | +23% |
| 0.98 | 10.0±7.0 | 9.1±5.6 | 7.0±4.2 | ✓+30% | 70% | 19.6±13.6 | 15.3±9.7 | +22% |

## 全体的な頑健性メモ

### FULL効果が頑健な設定（FULL−NC>+10% かつ win≥60%）
  - n_agents=8: FULL−NC=+30%  win=75%
  - n_agents=16: FULL−NC=+36%  win=95%
  - n_steps=1k: FULL−NC=+30%  win=75%
  - n_steps=2k: FULL−NC=+25%  win=75%
  - danger_radius=r=6: FULL−NC=+24%  win=60%
  - danger_radius=r=10: FULL−NC=+30%  win=75%
  - danger_radius=r=14: FULL−NC=+22%  win=60%
  - hear_sigma=σ=16: FULL−NC=+21%  win=70%
  - hear_sigma=σ=24: FULL−NC=+24%  win=60%
  - mark_strength=0.5×: FULL−NC=+20%  win=65%
  - mark_strength=1×: FULL−NC=+30%  win=75%
  - mark_strength=2×: FULL−NC=+30%  win=75%
  - avoid_decay=0.90: FULL−NC=+19%  win=65%
  - avoid_decay=0.95: FULL−NC=+30%  win=75%
  - avoid_decay=0.98: FULL−NC=+30%  win=70%

### FULL効果が消失・逆転する設定（FULL−NC<0% または win<40%）
  - n_steps=500: FULL−NC=+20%  win=30%

*生成: EchoLoop v6 sweep / 20 シード*
