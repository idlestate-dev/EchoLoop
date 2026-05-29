#!/usr/bin/env python3
"""
EchoLoop v6 — parameter sensitivity sweep (one-at-a-time, OAT).

For each of 6 parameters × 3 values:
  - run 20 seeds × 3 conditions (NO_CALLS / CALLS_NO_SPATIAL / FULL)
  - collect entries and exposure per seed

Produces:
  images/echoloop_v6_sweep.png       — 6×3 heatmap (entries + exposure)
  results/RESULTS6_sweep_ja.md       — detailed table
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import echoloop6 as ec

# ── Zone centre never moves ───────────────────────────────────────────────────
_DZ_CX, _DZ_CY = 40, 40

# ── Baseline (must match echoloop6.py defaults) ───────────────────────────────
BASELINE = dict(
    N_AGENTS=8,
    N_STEPS=1000,
    DANGER_ZONES=[(_DZ_CX, _DZ_CY, 10)],
    HEAR_SIGMA=35.0,
    AVOID_MARK_STRENGTH=0.18,
    AVOID_DECAY=0.95,
)

# ── Sweep spec: (row_label, ec_attr, [values], [cell_labels]) ─────────────────
# danger_radius is special: attr='DANGER_ZONES', value = the radius scalar
SWEEP = [
    ('n_agents',      'N_AGENTS',            [4,    8,    16  ], ['4',    '8',    '16'  ]),
    ('n_steps',       'N_STEPS',             [500,  1000, 2000], ['500',  '1k',   '2k'  ]),
    ('danger_radius', 'DANGER_ZONES',        [6,    10,   14  ], ['r=6',  'r=10', 'r=14']),
    ('hear_sigma',    'HEAR_SIGMA',          [8.0,  16.0, 24.0], ['σ=8',  'σ=16', 'σ=24']),
    ('mark_strength', 'AVOID_MARK_STRENGTH', [0.09, 0.18, 0.36], ['0.5×', '1×',   '2×'  ]),
    ('avoid_decay',   'AVOID_DECAY',         [0.90, 0.95, 0.98], ['0.90', '0.95', '0.98']),
]

N_SEEDS = 20
MODES   = ('NO_CALLS', 'CALLS_NO_SPATIAL', 'FULL')


# ── Module-level patching ─────────────────────────────────────────────────────
def _patch(attr, val):
    if attr == 'DANGER_ZONES':
        ec.DANGER_ZONES = [(_DZ_CX, _DZ_CY, int(val))]
    else:
        # Coerce to original type (int / float)
        setattr(ec, attr, type(getattr(ec, attr))(val))


def _restore():
    for attr, val in BASELINE.items():
        setattr(ec, attr, val)


# ── Single-config runner ──────────────────────────────────────────────────────
def run_config(attr, val, n_seeds):
    """
    Patch ec.{attr}=val, run n_seeds × 3 conditions, restore, return metrics.
    All other attrs stay at BASELINE.
    """
    _restore()
    _patch(attr, val)

    out = {m: defaultdict(list) for m in MODES}
    for seed in range(n_seeds):
        rng  = np.random.default_rng(seed)
        init = ec._make_initial_positions(rng)   # uses patched N_AGENTS / DANGER_ZONES
        for mode in MODES:
            agents, log, _ = ec.run_simulation(mode, init, seed=seed)
            out[mode]['entries'].append(sum(a.danger_entries  for a in agents))
            out[mode]['exposure'].append(sum(a.danger_exposure for a in agents))
    _restore()
    return out


# ── Full sweep ────────────────────────────────────────────────────────────────
def run_sweep(n_seeds=N_SEEDS):
    n_configs = sum(len(vals) for _, _, vals, _ in SWEEP)
    print(f"\nSensitivity sweep: {n_configs} configs × {n_seeds} seeds × 3 conditions "
          f"= {n_configs * n_seeds * 3} simulations\n")

    results = {}
    done = 0
    t0   = time.time()

    for label, attr, vals, clabs in SWEEP:
        for val, clab in zip(vals, clabs):
            done += 1
            print(f"  [{done:2d}/{n_configs}] {label}={clab:<6} ...",
                  end='  ', flush=True)
            t1  = time.time()
            res = run_config(attr, val, n_seeds)
            dt  = time.time() - t1
            nc  = float(np.mean(res['NO_CALLS']['entries']))
            fu  = float(np.mean(res['FULL']['entries']))
            pct = (nc - fu) / nc * 100 if nc > 0 else 0.0
            print(f"{dt:3.0f}s   NC={nc:.1f}  FULL={fu:.1f}  Δ={pct:+.1f}%")
            results[(label, clab)] = res

    print(f"\nTotal sweep: {time.time() - t0:.0f}s")
    return results


# ── Metric helpers ────────────────────────────────────────────────────────────
def _pct_impr(res, cond_ref, cond_test, metric):
    """% improvement of cond_test over cond_ref (positive = test is better)."""
    ref  = float(np.mean(res[cond_ref][metric]))
    test = float(np.mean(res[cond_test][metric]))
    return (ref - test) / ref * 100 if ref > 0 else 0.0


def _win_rate(res, cond_ref, cond_test, metric):
    """Fraction of seeds where cond_test < cond_ref."""
    return float(np.mean(
        np.array(res[cond_test][metric]) < np.array(res[cond_ref][metric])))


def _mean_std(res, cond, metric):
    v = res[cond][metric]
    return float(np.mean(v)), float(np.std(v))


# ── Heatmap ───────────────────────────────────────────────────────────────────
def plot_heatmap(results, out_path):
    """
    Two 6×3 heatmaps (entries / exposure).
    Rows = parameters, columns = values.
    Color = FULL vs NO_CALLS % improvement (green = FULL better).
    Cell text = value_label / pct / win_rate / CNS_marker.
    """
    _BG = '#1a1a2e'
    n_r, n_c = len(SWEEP), 3

    fig, axes = plt.subplots(1, 2, figsize=(17, 7), facecolor=_BG)
    fig.suptitle(
        'EchoLoop v6 — Sensitivity Sweep  |  FULL vs NO_CALLS % improvement '
        '(green = FULL reduces metric)\n'
        'Cell: value / FULL−NC% / win=fraction of seeds FULL<NC / CNS: △>5% △—neutral ▽<−5%',
        color='white', fontsize=10, y=1.02)

    row_labels = [s[0] for s in SWEEP]
    metrics    = ('entries', 'exposure')

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        ax.set_facecolor(_BG)

        data  = np.zeros((n_r, n_c))
        annot = [['' for _ in range(n_c)] for _ in range(n_r)]

        for pi, (label, attr, vals, clabs) in enumerate(SWEEP):
            for vi, (val, clab) in enumerate(zip(vals, clabs)):
                res  = results[(label, clab)]
                pct  = _pct_impr(res, 'NO_CALLS', 'FULL',             metric)
                win  = _win_rate(res, 'NO_CALLS', 'FULL',             metric)
                cpct = _pct_impr(res, 'NO_CALLS', 'CALLS_NO_SPATIAL', metric)

                data[pi, vi] = pct

                cns = '△' if cpct > 5 else ('▽' if cpct < -5 else '—')
                annot[pi][vi] = (
                    f"{clab}\n"
                    f"{pct:+.0f}%\n"
                    f"win={win:.0%}  CNS{cns}"
                )

        vmax = max(30.0, float(np.abs(data).max()))
        im   = ax.imshow(data, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                         aspect='auto', interpolation='nearest')
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors='#aaaacc')
        cb.set_label('% improvement', color='#aaaacc', fontsize=7)

        ax.set_xticks([])
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(row_labels, color='white', fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a5a')
        ax.tick_params(colors='#667799')
        ax.set_title(f'metric: {metric}', color='white', fontsize=10, pad=8)

        # Grid lines between cells
        for x in np.arange(-0.5, n_c, 1):
            ax.axvline(x, color='#2a2a5a', lw=0.8)
        for y in np.arange(-0.5, n_r, 1):
            ax.axhline(y, color='#2a2a5a', lw=0.8)

        for pi in range(n_r):
            for vi in range(n_c):
                pct_val    = data[pi, vi]
                txt_color  = 'white' if abs(pct_val) > vmax * 0.45 else '#dddddd'
                ax.text(vi, pi, annot[pi][vi],
                        ha='center', va='center', fontsize=6.5,
                        color=txt_color, linespacing=1.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    print(f"Saved → {out_path}")


# ── Markdown report ───────────────────────────────────────────────────────────
def write_markdown(results, n_seeds, out_path):
    lines = [
        '# EchoLoop v6 — パラメータ感度スイープ結果',
        '',
        f'一変数一変動（OAT）スイープ。各設定 {n_seeds} シード × 3条件。',
        '他パラメータはすべてベースライン値に固定。',
        '',
        '**列の説明**',
        '- NC/CNS/FULL = 平均±標準偏差（侵入回数）',
        '- FULL−NC % = FULLの侵入回数がNO_CALLSより低下した割合（正 = FULL優位）',
        '- ✓win = FULLがNO_CALLSを下回ったシードの割合',
        '- CNS% = CNSのNO_CALLS比改善率　△>+5%  ▽<−5%  — = ±5%以内',
        '',
    ]

    for label, attr, vals, clabs in SWEEP:
        lines += [
            f'## {label}',
            '',
            '| 値 | NC entries | CNS entries | FULL entries |'
            ' FULL−NC% | ✓win | NC exposure | FULL exposure | exp FULL−NC% |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for val, clab in zip(vals, clabs):
            res = results[(label, clab)]

            nc_em, nc_es   = _mean_std(res, 'NO_CALLS',         'entries')
            cn_em, cn_es   = _mean_std(res, 'CALLS_NO_SPATIAL', 'entries')
            fu_em, fu_es   = _mean_std(res, 'FULL',             'entries')
            nc_xm, nc_xs   = _mean_std(res, 'NO_CALLS',         'exposure')
            fu_xm, fu_xs   = _mean_std(res, 'FULL',             'exposure')

            pct_e  = _pct_impr(res, 'NO_CALLS', 'FULL',             'entries')
            pct_cn = _pct_impr(res, 'NO_CALLS', 'CALLS_NO_SPATIAL', 'entries')
            win_e  = _win_rate(res, 'NO_CALLS', 'FULL',             'entries')
            pct_x  = _pct_impr(res, 'NO_CALLS', 'FULL',             'exposure')

            cns_mark = '△' if pct_cn > 5 else ('▽' if pct_cn < -5 else '—')
            beat     = '✓' if pct_e > 0 else '✗'

            lines.append(
                f'| {clab} '
                f'| {nc_em:.1f}±{nc_es:.1f} '
                f'| {cn_em:.1f}±{cn_es:.1f} '
                f'| {fu_em:.1f}±{fu_es:.1f} '
                f'| {beat}{pct_e:+.0f}% '
                f'| {win_e:.0%} '
                f'| {nc_xm:.1f}±{nc_xs:.1f} '
                f'| {fu_xm:.1f}±{fu_xs:.1f} '
                f'| {pct_x:+.0f}% |'
            )
        lines.append('')

    # Robustness summary
    lines += [
        '## 全体的な頑健性メモ',
        '',
    ]
    robust_full, fragile_full = [], []
    for label, attr, vals, clabs in SWEEP:
        for val, clab in zip(vals, clabs):
            res = results[(label, clab)]
            pct = _pct_impr(res, 'NO_CALLS', 'FULL', 'entries')
            win = _win_rate(res, 'NO_CALLS', 'FULL', 'entries')
            tag = f'{label}={clab}'
            if pct > 10 and win >= 0.60:
                robust_full.append(f'  - {tag}: FULL−NC={pct:+.0f}%  win={win:.0%}')
            elif pct < 0 or win < 0.40:
                fragile_full.append(f'  - {tag}: FULL−NC={pct:+.0f}%  win={win:.0%}')

    lines.append('### FULL効果が頑健な設定（FULL−NC>+10% かつ win≥60%）')
    lines += (robust_full if robust_full else ['  （なし）'])
    lines.append('')
    lines.append('### FULL効果が消失・逆転する設定（FULL−NC<0% または win<40%）')
    lines += (fragile_full if fragile_full else ['  （なし）'])
    lines += [
        '',
        f'*生成: EchoLoop v6 sweep / {n_seeds} シード*',
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    results = run_sweep(N_SEEDS)

    plot_heatmap(results,
                 os.path.join(repo_root, 'images', 'echoloop_v6_sweep.png'))
    write_markdown(results, N_SEEDS,
                   os.path.join(repo_root, 'results', 'RESULTS6_sweep_ja.md'))


if __name__ == '__main__':
    main()
