import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_rolling_mean(arr, window):
    cum_sum = np.cumsum(arr)
    roll_mean = np.array(arr, dtype=float) 
    for i in range(1, window):
        roll_mean[i] = cum_sum[i] / (i + 1)
    roll_mean[window:] = (cum_sum[window:] - cum_sum[:-window]) / window
    return roll_mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, required=True, help="Suffix after Policy (e.g., K100_T1000000)")
    parser.add_argument("--target", type=str, required=True, help="'single', 'coop', or m_value (e.g., '20')")
    parser.add_argument("--zoom", type=float, nargs='?', const=80.0, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    policies = ["UCB", "Epsilon-Greedy", "Thompson-Sampling"]
    colors = {"UCB": "tab:blue", "Epsilon-Greedy": "tab:orange", "Thompson-Sampling": "tab:green"}
    
    extracted_data = {}
    t_steps = None
    k_arms = "Unknown"

    for policy in policies:
        # ---> PERBAIKAN DI SINI: MENAMBAHKAN 'ConstantTotal_' <---
        filename = f"mab_results_ConstantTotal_{policy}_{args.suffix}.npz"
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: '{filename}' not found.")
            continue
            
        npz_data = np.load(filepath, allow_pickle=True)
        if t_steps is None:
            t_steps = npz_data['t_steps']
            k_arms = npz_data.get('k_arms', 'Unknown')
            
        if args.target == 'single':
            reg_key, rew_key, title_target = 'reg_single', 'rew_single', "Single-Agent Baseline"
        elif args.target == 'coop':
            reg_key, rew_key, title_target = 'reg_coop', 'rew_coop', "Fully Coop (0% Malicious)"
        else:
            m_val = int(args.target)
            n_val = 25 - m_val
            pct = int((m_val / 25) * 100)
            reg_key = f'reg_mal_n{n_val}_m{m_val}'
            rew_key = f'rew_mal_n{n_val}_m{m_val}'
            title_target = f"Malicious Attack ({pct}% Malicious)"
            
        if reg_key in npz_data.files and rew_key in npz_data.files:
            extracted_data[policy] = {'reg': npz_data[reg_key], 'rew': npz_data[rew_key]}

    if not extracted_data:
        print("Error: No data found for the requested target.")
        return

    mark_interval = max(1, int(len(t_steps) / 10))
    window_size = min(50000, len(t_steps) // 10)
    save_suffix = f"COMPARE_{args.target}_{args.suffix}"
    if args.zoom is not None: save_suffix += f"_zoom{int(args.zoom)}"

    os.makedirs(results_dir, exist_ok=True)

    # Plot Regret
    plt.figure(figsize=(9, 5.5), dpi=150)
    for pol, data in extracted_data.items():
        plt.plot(t_steps, data['reg'], label=f"{pol}", color=colors[pol], linewidth=1.5, marker='|', markevery=mark_interval, markersize=10)
    plt.title(f"Policy Comparison: {title_target} | K={k_arms}")
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Cumulative Regret")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_1_regret_{save_suffix}.png"))

    print(f"\nCross-Policy Comparison Plots saved to results/ with suffix '{save_suffix}'")
    if not args.no_show: plt.show()

if __name__ == "__main__":
    main()