import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from src.env import load_movielens_arms
from src.simulation import run_single_agent_baseline, run_multi_agent_system

def get_rolling_mean(arr, window):
    cum_sum = np.cumsum(arr)
    roll_mean = np.array(arr, dtype=float)
    for i in range(1, window):
        roll_mean[i] = cum_sum[i] / (i + 1)
    roll_mean[window:] = (cum_sum[window:] - cum_sum[:-window]) / window
    return roll_mean

def plot_results(t_steps, results_dict, k_arms, policy, run_suffix, show_plot=True):
    regret = results_dict['regret']
    reward = results_dict['reward']
    
    mark_interval = max(1, int(len(t_steps) / 10))
    window_size = min(50000, len(t_steps) // 10)
    mal_colors = ['tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']

    os.makedirs("results", exist_ok=True)

    # ==========================================
    # PLOT 1: Cumulative Regret
    # ==========================================
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(t_steps, regret['single'], label="Single-Agent Baseline", linestyle='--', color='tab:blue', marker='|', markevery=mark_interval, markersize=10)
    plt.plot(t_steps, regret['coop'], label="Fully Coop. (0% Malicious)", linestyle=':', color='tab:red', marker='|', markevery=mark_interval, markersize=10)
    
    mal_keys = [k for k in regret.keys() if k.startswith('mal_')]
    mal_keys = sorted(mal_keys, key=lambda x: int(x.split('_')[2].replace('m', '')))
    
    for c_idx, key in enumerate(mal_keys):
        n_val = int(key.split('_')[1].replace('n', ''))
        m_val = int(key.split('_')[2].replace('m', ''))
        pct = int((m_val / (n_val + m_val)) * 100)
        plt.plot(t_steps, regret[key], label=f"Malicious Attack ({pct}% Malicious)", linestyle='-.', color=mal_colors[c_idx % len(mal_colors)], marker='|', markevery=mark_interval, markersize=10)

    plt.title(f"Cumulative Regret (Total 25 Agents) | {policy} - K={k_arms}")
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Cumulative Regret ($\frac{1}{n} \sum_{i=1}^n R_T^{(i)}$)")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"plot_1_regret_{run_suffix}.png"))

    # ==========================================
    # PLOT 2: Rolling Mean Reward
    # ==========================================
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(t_steps, get_rolling_mean(reward['single'], window_size), label="Single-Agent Baseline", linestyle='--', color='tab:blue')
    plt.plot(t_steps, get_rolling_mean(reward['coop'], window_size), label="Fully Coop. (0% Malicious)", linestyle=':', color='tab:red')
    
    for c_idx, key in enumerate(mal_keys):
        n_val = int(key.split('_')[1].replace('n', ''))
        m_val = int(key.split('_')[2].replace('m', ''))
        pct = int((m_val / (n_val + m_val)) * 100)
        plt.plot(t_steps, get_rolling_mean(reward[key], window_size), label=f"Malicious Attack ({pct}% Malicious)", linestyle='-.', color=mal_colors[c_idx % len(mal_colors)])

    plt.title(f"{window_size}-Round Rolling Mean (Total 25 Agents) | {policy} - K={k_arms}")
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Reward")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"plot_3_rolling_reward_{run_suffix}.png"))

    if show_plot: plt.show()
    else: plt.close('all')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/ml-20m/ratings.csv")
    parser.add_argument("-k", "--k_arms", type=int, default=1000)
    parser.add_argument("-t", "--t_horizon", type=int, default=None)
    parser.add_argument("--policy", type=str, default="UCB", choices=["UCB", "Epsilon-Greedy", "Thompson-Sampling"])
    parser.add_argument("--explore_param", type=float, default=None)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    DATA_PATH = args.data
    K_ARMS = args.k_arms
    T = args.t_horizon if args.t_horizon else K_ARMS * 250
    POLICY = args.policy
    BETA = args.beta
    SHOW_PLOT = not args.no_show
    
    if args.explore_param is None:
        EXPLORE_PARAM = 1.5 if POLICY == "UCB" else (20000.0 if POLICY == "Epsilon-Greedy" else 1.0)
    else:
        EXPLORE_PARAM = args.explore_param

    # ---> VARIABEL BARU (MEANS, STDS) MENGGANTIKAN (ARMS, TRUE_MEANS) <---
    means, stds = load_movielens_arms(DATA_PATH, min_ratings=100)
    actual_k = min(K_ARMS, len(means))
    means = means[:actual_k]
    stds = stds[:actual_k]
    t_steps = np.arange(1, T + 1)
    
    run_suffix = f"ConstantTotal_{POLICY}_K{actual_k}_T{T}"
    scenarios = [(20, 5), (15, 10), (10, 15), (5, 20)] # Constant Total 25
    
    print(f"\n=============================================")
    print(f"   STARTING MAB SIMULATION (TOTAL 25 AGENTS)")
    print(f"   Policy: {POLICY} | K={actual_k} | T={T}")
    print(f"=============================================\n")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        # ---> UPDATE: MEMASUKKAN MEANS DAN STDS KE FUNGSI SIMULASI <---
        f_single = executor.submit(run_single_agent_baseline, means, stds, T, POLICY, EXPLORE_PARAM, 0)
        f_coop = executor.submit(run_multi_agent_system, means, stds, T, 25, 0, BETA, POLICY, EXPLORE_PARAM, 1)
        
        f_mal_dict = {}
        for idx, (n_h, m_m) in enumerate(scenarios):
            f_mal_dict[f"n{n_h}_m{m_m}"] = executor.submit(run_multi_agent_system, means, stds, T, n_h, m_m, BETA, POLICY, EXPLORE_PARAM, 2 + idx)
        
        reg_single, rew_single = f_single.result()
        reg_coop, rew_coop = f_coop.result()
        
        mal_results = {}
        for key, future in f_mal_dict.items():
            mal_results[key] = future.result() 

    print("\nSimulations finished! Saving data...")
    os.makedirs("results", exist_ok=True)
    
    save_data = {'t_steps': t_steps, 'k_arms': actual_k, 'policy': POLICY, 'reg_single': reg_single, 'rew_single': rew_single, 'reg_coop': reg_coop, 'rew_coop': rew_coop}
    for key, (reg, rew) in mal_results.items():
        save_data[f'reg_mal_{key}'] = reg
        save_data[f'rew_mal_{key}'] = rew
        
    np.savez(os.path.join("results", f"mab_results_{run_suffix}.npz"), **save_data)

    regret_dict = {'single': reg_single, 'coop': reg_coop}
    reward_dict = {'single': rew_single, 'coop': rew_coop}
    for key, (reg, rew) in mal_results.items():
        regret_dict[f'mal_{key}'] = reg
        reward_dict[f'mal_{key}'] = rew
    
    plot_results(t_steps, {'regret': regret_dict, 'reward': reward_dict}, actual_k, POLICY, run_suffix, show_plot=SHOW_PLOT)
    print(f"Data saved successfully with suffix '{run_suffix}'!")

if __name__ == "__main__":
    main()