"""run_all.py — Entry point for EchoLoop co-evolution experiments.

Run sessions in order. Earlier sessions are commented out (results already saved).
Uncomment individual blocks to re-run specific experiments.
"""
import os

os.makedirs('images', exist_ok=True)


# ─── Early explorations ───────────────────────────────────────────────────────
# from explorations import (
#     run_input_comparison, plot_input_comparison, plot_topology_fingerprint,
#     run_encoding_experiment, plot_encoding,
#     run_stdp_encoding_experiment,
#     run_delta_comparison, plot_delta_comparison,
#     run_convergence_test, plot_convergence_test,
#     run_cycle_analysis, plot_cycle_analysis,
#     run_ei_comparison,
#     run_inhibition_sweep, plot_inhibition_sweep,
#     run_threshold_sweep, plot_threshold_sweep,
#     run_silencing_sweep, plot_silencing_sweep,
#     run_isolation_sweep, plot_isolation_sweep,
#     run_input_removal_test, plot_input_removal,
#     run_fixed_topology_ei_test, plot_fixed_topology_control,
#     run_pattern_encoding_experiment, plot_pattern_encoding, plot_pattern_encoding_training,
#     run_pattern_encoding_experiment_v2, plot_pattern_encoding_v2,
#     plot_pattern_encoding_training_v2, plot_pattern_coexistence,
#     run_loop_resonance_experiment, plot_loop_resonance,
#     run_loop_combination_experiment, plot_loop_combination,
#     run_association_experiment, plot_association,
#     run_world_association_experiment, plot_world_association,
# )

# results = run_input_comparison(K=20)
# plot_input_comparison(results)
# plot_topology_fingerprint(results)

# enc = run_encoding_experiment()
# plot_encoding(enc)

# run_stdp_encoding_experiment()

# delta_results = run_delta_comparison([1, 5, 10, 20, 30, 50])
# plot_delta_comparison(delta_results, [1, 5, 10, 20, 30, 50])

# conv_results = run_convergence_test()
# plot_convergence_test(conv_results)

# cycle_snapshots = run_cycle_analysis()
# plot_cycle_analysis(cycle_snapshots)

# run_ei_comparison()

# sweep = run_inhibition_sweep()
# plot_inhibition_sweep(sweep)

# threshold_sweep = run_threshold_sweep()
# plot_threshold_sweep(threshold_sweep)

# silencing_sweep = run_silencing_sweep()
# plot_silencing_sweep(silencing_sweep)

# isolation_sweep = run_isolation_sweep()
# plot_isolation_sweep(isolation_sweep)

# removal_data = run_input_removal_test()
# plot_input_removal(removal_data)

# res_A, res_B = run_fixed_topology_ei_test()
# plot_fixed_topology_control(res_A, res_B)

# enc_data = run_pattern_encoding_experiment()
# plot_pattern_encoding(enc_data)
# plot_pattern_encoding_training(enc_data)

# enc_data_v2 = run_pattern_encoding_experiment_v2()
# plot_pattern_encoding_v2(enc_data_v2)
# plot_pattern_encoding_training_v2(enc_data_v2)
# plot_pattern_coexistence(enc_data_v2)

# resonance_data = run_loop_resonance_experiment()
# plot_loop_resonance(resonance_data)

# combination_data = run_loop_combination_experiment()
# plot_loop_combination(combination_data)

# association_data = run_association_experiment()
# plot_association(association_data)

# assoc_data = run_world_association_experiment()
# plot_world_association(assoc_data)


# ─── Session 3: E/I threshold evolution ──────────────────────────────────────
# from session_3_ei_evo import (
#     run_ei_threshold_evolution, plot_ei_evolution,
#     run_sparse_association, plot_sparse_association,
#     run_sparse_comparison, plot_sparse_comparison,
# )

# ei_evo_data = run_ei_threshold_evolution(
#     n_agents=10, n_generations=30, n_survivors=3,
#     n_episodes_per_agent=3, N=20, K=10, temperature=1.0, seed=42)
# plot_ei_evolution(ei_evo_data)
# sparse_assoc = run_sparse_association(ei_evo_data['best_genome'], seed=42)
# plot_sparse_association(sparse_assoc)
# comparison = run_sparse_comparison(ei_evo_data['best_genome'], seed=42)
# plot_sparse_comparison(comparison)


# ─── Session 4: Context interference environment ─────────────────────────────
# from session_4_context import (
#     run_context_ei_evolution, plot_context_ei_evolution,
#     run_context_comparison, plot_context_comparison,
# )
# from session_3_ei_evo import run_sparse_association, plot_sparse_association

# _simple_genome = {'ei_threshold': 0.9500, 'recovery_ratio': 0.2563, 'recovery_delay': 22}
# context_evo_data = run_context_ei_evolution(
#     n_agents=10, n_generations=30, n_survivors=3,
#     n_episodes_per_agent=3, N=20, K=10, temperature=1.0, seed=42)
# plot_context_ei_evolution(context_evo_data, simple_data=None)
# context_sparse = run_sparse_association(context_evo_data['best_genome'], seed=42)
# plot_sparse_association(
#     context_sparse, fname='images/session_4/results_context_sparse_association.png')
# context_comp = run_context_comparison(
#     _simple_genome, context_evo_data['best_genome'], seed=42)
# plot_context_comparison(context_comp)


# ─── Session 5: Association parameter sweep ───────────────────────────────────
# from session_5_sweep import (
#     run_association_sweep, plot_association_sweep, plot_association_probe,
# )

# sweep_data = run_association_sweep(N=20, seed=42, K=5, T_phase=500, T_probe=200)
# plot_association_sweep(sweep_data)
# plot_association_probe(sweep_data)


# ─── Session 6: Dynamic E/I vs static inhibition ─────────────────────────────
# from session_6_ei_static import (
#     run_ei_vs_static_experiments, plot_ei_vs_static_overwrite,
#     run_ei_vs_static_context, plot_ei_vs_static_context,
# )

# s6_ab = run_ei_vs_static_experiments(N=20, seed=42, K=5, T_probe=100, probe_interval=10)
# plot_ei_vs_static_overwrite(s6_ab)
# s6_c = run_ei_vs_static_context(s6_ab, N=20, seed=42, n_episodes=10, T_episode=100)
# plot_ei_vs_static_context(s6_c)


# ─── Session 7: Context-dependent activation patterns ────────────────────────
from session_7_context_activation import (
    run_context_activation_experiment, plot_context_activation, plot_context_topology,
    run_context_control_experiment, plot_context_control,
)

s7_data = run_context_activation_experiment(N=20, seed=42)
plot_context_activation(s7_data)
plot_context_topology(s7_data)

def _s7_score(k):
    p = s7_data['sweep'][k]
    if p['norm_A'] < 1e-4 or p['norm_B'] < 1e-4:
        return -1.0
    return p['cos_dist'] if p['cos_dist'] == p['cos_dist'] else 0.0

s7_best = max(s7_data['sweep'].keys(), key=_s7_score)
bp = s7_data['sweep'][s7_best]
print(f'Best (non-degenerate) condition: T_phase={s7_best[0]}, '
      f'switch_interval={s7_best[1]}, '
      f'cos_dist={bp["cos_dist"]:.4f}  '
      f'‖A‖={bp["norm_A"]:.4f}  ‖B‖={bp["norm_B"]:.4f}')

s7_ctrl = run_context_control_experiment(s7_best, N=20, seed=42)
plot_context_control(s7_data, s7_ctrl, s7_best)


# ─── Session 8: 世界に問う ────────────────────────────────────────────────────
# from session_8_world_test import run_world_test_experiment, plot_world_test
# s8_data = run_world_test_experiment(N=20, seed=42, n_agents=10, n_episodes=20)
# plot_world_test(s8_data)


# ─── Session 9: 世界がトポロジーを彫刻するメカニズム ──────────────────────────
# from session_9_topology_sculpting import (
#     run_topology_convergence, plot_topology_convergence,
#     run_survivor_topology,    plot_survivor_topology,
#     run_experience_trace,     plot_experience_trace,
# )
# s9_expA = run_topology_convergence(seed=42, T_total=5000, snapshot_interval=500)
# plot_topology_convergence(s9_expA)
# s9_expB = run_survivor_topology(seed=42, n_agents=20)
# plot_survivor_topology(s9_expB)
# s9_expC = run_experience_trace(s9_expB)
# plot_experience_trace(s9_expC)


# ─── Session 10: アウトプットノードによる身体化 ───────────────────────────────
# from session_10_embodied_output import (
#     run_single_agent,    plot_single_agent,
#     run_evolution,       plot_evolution,
#     run_action_patterns, plot_action_patterns,
# )
# s10_expA = run_single_agent(seed=42, T_total=10000, window=1000)
# plot_single_agent(s10_expA)
# s10_expB = run_evolution(seed=42, n_gen=50, n_agents=10, n_ep=5, n_surv=3)
# plot_evolution(s10_expB)
# s10_expC = run_action_patterns(s10_expB['new']['best_G'], s10_expB['new']['best_W'],
#                                seed=42, n_ep=30)
# plot_action_patterns(s10_expC)


# ─── Session 11: 自発的ノイズによる局所最適からの脱出 ──────────────────────────
# from session_11_noise_escape import (
#     run_noise_sweep, plot_noise_sweep,
#     run_evolution as run_s11_evolution, plot_evolution as plot_s11_evolution,
# )
# s11_expA = run_noise_sweep(seed=42)
# plot_noise_sweep(s11_expA)
# s11_expB = run_s11_evolution(seed=42, best_noise=s11_expA['best_condition'])
# plot_s11_evolution(s11_expB)


# ─── Session 12: 探索と記憶の固定（睡眠仮説）────────────────────────────────────
from session_12_sleep_consolidation import (
    run_consolidation_sweep, plot_consolidation_sweep,
    run_evolution as run_s12_evolution, plot_evolution as plot_s12_evolution,
)

s12_expA = run_consolidation_sweep(seed=42)
plot_consolidation_sweep(s12_expA)

s12_expB = run_s12_evolution(seed=42, best_T=s12_expA['best_T'])
plot_s12_evolution(s12_expB)
