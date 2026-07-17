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

    for policy in policies:
        filename = f"mab_results_ConstantTotal_{policy}_{args.suffix}.npz"
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: '{filename}' not found.")
            continue
            
        npz_data = np.load(filepath, allow_pickle=True)
        if t_steps is None:
            t_steps = npz_data['t_steps']
            
        if args.target == 'single':
            reg_key, rew_key = 'reg_single', 'rew_single'
        elif args.target == 'coop':
            reg_key, rew_key = 'reg_coop', 'rew_coop'
        else:
            m_val = int(args.target)
            n_val = 25 - m_val
            reg_key = f'reg_mal_n{n_val}_m{m_val}'
            rew_key = f'rew_mal_n{n_val}_m{m_val}'
            
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

    # ==========================================
    # PLOT 1: Regret
    # ==========================================
    plt.figure(figsize=(9, 5.5), dpi=150)
    for pol, data in extracted_data.items():
        plt.plot(t_steps, data['reg'], label=f"{pol}", color=colors[pol], linewidth=1.5, marker='|', markevery=mark_interval, markersize=10)
    
    plt.title("Policy Comparison") # ---> JUDUL DIPERBARUI <---
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Cumulative Regret")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_1_regret_{save_suffix}.png"))

    # ==========================================
    # PLOT 2: Rolling Mean
    # ==========================================
    plt.figure(figsize=(9, 5.5), dpi=150)
    rm_dict = {}
    for pol, data in extracted_data.items():
        rm_dict[pol] = get_rolling_mean(data['rew'], window_size)
        plt.plot(t_steps, rm_dict[pol], label=f"{pol}", color=colors[pol])
    
    plt.title("Policy Comparison") # ---> JUDUL DIPERBARUI <---
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Reward")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    if args.zoom is not None:
        zoom_pct = max(0.0, min(99.9, args.zoom))
        cutoff = int(len(t_steps) * (zoom_pct / 100.0))
        plt.xlim(t_steps[cutoff], t_steps[-1])
        y_min, y_max = float('inf'), float('-inf')
        for rm_arr in rm_dict.values():
            tail_data = rm_arr[cutoff:]
            y_min = min(y_min, np.min(tail_data))
            y_max = max(y_max, np.max(tail_data))
        if y_min != float('inf'):
            margin = (y_max - y_min) * 0.1 
            plt.ylim(y_min - margin if margin != 0 else y_min - 0.05, y_max + margin if margin != 0 else y_max + 0.05)
            
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_3_rolling_reward_{save_suffix}.png"))

    print(f"\nCross-Policy Comparison Plots saved to results/ with suffix '{save_suffix}'")
    if not args.no_show: plt.show()

if __name__ == "__main__":
    main()