# Co-Evolution Sim — Experiment Series

**Script:** `echo_sim.py`
**Branch:** `feature/co-evolution-sim`
**Theme:** 活動とトポロジーの相互進化（co-evolution）による自己組織化ダイナミクスの探索

---

## 実験系列の概要

20ノードの有向ネットワークを舞台に、ノード活動（`tanh` 伝播）とトポロジー（Hebbian強化 + 指数減衰）が互いに影響しあう系を段階的に発展させた。各実験は直前の結果から生じた問いに答える形で連鎖している。

---

## Exp 1 — K 比較：活動更新速度とトポロジー変化速度の比

**問い:** K（トポロジー更新間隔）を変えると最終トポロジーはどう変わるか？

**結果:** K が小さいほど（活動とトポロジーの時定数が近いほど）エッジ数が多く、クラスタリング係数が高い。K=1 では密なネットワークが、K=50 では疎なネットワークが形成される。

**図:** `images/results_comparison_fine.png`

---

## Exp 2 — 入力パターン比較：constant / alternating / random

**問い:** 同じ初期状態から異なる入力パターンを与えると、最終トポロジーは異なるか？

**結果:** constant と alternating で異なるトポロジーが形成される。入力パターンがトポロジーに痕跡を残す。

**図:** `images/results_input_comparison.png`, `images/results_topology_fingerprint.png`

---

## Exp 3 — 入力履歴の符号化：topology fingerprint

**問い:** 訓練後のトポロジーは入力履歴を符号化しているか？沈黙期後のプローブ応答で識別できるか？

**結果:** A（constant）と B（alternating）で訓練したネットワークのプローブ軌道間のユークリッド距離が 1.0 を超え、トポロジーが入力履歴を保持していることを確認。

**図:** `images/results_encoding.png`

---

## Exp 4 — STDP：タイミング依存可塑性

**問い:** 率ベースの強化則を STDP（スパイクタイミング依存）に置き換えると何が変わるか？

**結果:** STDP でも符号化は成立するが、沈黙期に全エッジが崩壊（full -1 抑制によるトポロジー消失）。エッジ崩壊は finding として記録。

**図:** `images/results_stdp_comparison.png`, `images/results_stdp_encoding.png`

---

## Exp 5 — トポロジー変化量の追跡

**問い:** 各トポロジー更新ステップでの変化量はどう推移するか？K によって違いはあるか？

**結果:** K=1 では初期に大きな変化があり急速に収束。K=50 はゆっくりと変化が続く。

**図:** `images/results_delta_comparison.png`

---

## Exp 6 — 長期収束テスト

**問い:** K=50 は長期的に K=1 と同じ構造に収束するか？

**結果:** 収束せず、K の違いが質的に異なる最終状態を生む。K は最終トポロジーの構造クラスを決定する。

**図:** `images/results_convergence.png`

---

## Exp 7 — サイクル検出

**問い:** 共進化は自発的に有向サイクル（ループ）を生成するか？

**結果:** K=1 では T=100 から >10,000 サイクルが発生。SCC は初期から大規模（19〜20ノード）。K=50 ではサイクルが少なく構造が疎。

**図:** `images/results_cycle_analysis.png`

---

## Exp 8 — E/I ダイナミクス（率ベース抑制）

**問い:** 過活性ノードが inhibitory に転換する仕組みを加えると、活動の爆発を防げるか？

**結果:** `tanh(max(0, influence))` の半波整流により inhibitory の影響がゼロになる問題を発見。inhibition_strength の値によらず結果が同一。finding として記録し次の実験へ。

**図:** `images/results_ei_comparison.png`, `images/results_ei_inhibition.png`, `images/results_ei_threshold.png`

---

## Exp 9 — E/I silencing：抑制ではなく凍結

**問い:** inhibitory ノードが活動を減衰させるのではなく、関連エッジを「凍結」させる設計にすると何が起きるか？

**設計変更:** inhibitory ノード関連エッジは強化・減衰・削除の対象外。重みは現在値に固定。活動伝播は全エッジで維持（type_strength なし）。

**結果:** inhibitory ノードの活動が 0.95〜0.99 と高止まり。inhibitory→excitatory の復帰が起きない。エッジの大半が silenced に固定される。

**図:** `images/results_ei_silencing.png`

---

## Exp 10 — E/I isolation：完全切断

**問い:** silenced エッジを活動伝播からも切断し、inhibitory ノードを指数減衰させると何が起きるか？

**設計変更:** 活動伝播は exc-exc エッジのみ。inhibitory ノード: `a(t+1) = a(t) * 0.9`。

**結果:** inhibitory ノードの活動がゼロに収束（0.001〜0.004）。ゼロになることで `recent_mean < threshold*0.5` を満たし、inh→exc の復帰が頻発（500回/5000ステップ）。ei_threshold=0.9 のみが質的に異なる状態（active edges 218、clustering 0.92、variance 0.18）に落ち着く。

**図:** `images/results_ei_isolation.png`

---

## Exp 11 — 入力除去テスト

**問い:** 入力ありで訓練した後、入力を除去しても活動は維持されるか？

**プロトコル:** Training（T=3000, input=0.5）→ No input（T=3000, input=0.0）→ Restored（T=2000, input=0.5）

**結果 (ei_threshold=0.9):**
- Training 終了: variance=0.117, active_edges=22, excitatory=9
- No-input 終了: variance=0.157, active_edges=249, excitatory=20 → **activity_maintained: True**
- Restored 終了: variance=0.024, active_edges=0, excitatory=4

**逆説的発見:** 入力除去後に活動が増加（variance 上昇）。input=0 により inh ノードが一斉に exc へ復帰し、active edges が急増。入力復元後に再崩壊。

**図:** `images/results_input_removal.png`

---

## Exp 12 — 固定トポロジー対照実験

**問い:** 振動はトポロジーの共進化を必要とするか？E/I スイッチングルールだけで振動が生成されるか？

**プロトコル:** T=3000 で訓練 → 訓練済みトポロジーを出発点に A（動的トポロジー）と B（固定トポロジー）を T=5000 実行（input なし）。

**結果:**
| 条件 | variance mean (last 1000) | oscillation |
|---|---|---|
| Dynamic topology | 0.1050 | True |
| Fixed topology | 0.1070 | True |

**結論:** 振動はトポロジーの共進化を必要としない。E/I スイッチングルールだけで自律的振動が維持される。訓練済みネットワーク構造が振動を可能にする基盤となっている。

**図:** `images/results_fixed_topology_control.png`

---

## 全体を通じた主な発見

1. **K がトポロジークラスを決定する:** K=1（密・高クラスタ・多サイクル）と K=50（疎・低クラスタ）は長期的にも収束しない。
2. **トポロジーは入力履歴を符号化する:** 訓練後の重み行列がプローブ応答の違いとして現れる。
3. **inhibitory 設計の難しさ:** 率ベース抑制（半波整流）は inhibitory の効果をゼロにする。完全切断（isolation）で初めて機能するが、今度は過剰な復帰サイクルが生じる。
4. **振動の起源は E/I スイッチング:** トポロジー変化は振動の必要条件ではない。初期構造さえ適切なら E/I スイッチングルールだけで自律振動が維持される。
5. **入力除去で活動が増加する逆説:** 入力ゼロにより inhibitory ノードが一斉回復し、却って excitatory-excitatory ネットワークが活性化する。
