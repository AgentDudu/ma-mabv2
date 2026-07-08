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
    parser.add_argument("--suffix", type=str, required=True, help="Suffix after Policy (e.g., K8546_T3500000)")
    parser.add_argument("--zoom", type=float, nargs='?', const=80.0, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    policies = ["UCB", "Epsilon-Greedy", "Thompson-Sampling"]
    colors = {"UCB": "tab:blue", "Epsilon-Greedy": "tab:orange", "Thompson-Sampling": "tab:green"}
    
    data_all = {}
    t_steps = None
    k_arms = "Unknown"

    for policy in policies:
        filename = f"mab_results_{policy}_{args.suffix}.npz"
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        npz_data = np.load(filepath, allow_pickle=True)
        if t_steps is None:
            t_steps = npz_data['t_steps']
            k_arms = npz_data.get('k_arms', 'Unknown')
            
        data_all[policy] = {
            'reg_single': npz_data['reg_single'],
            'rew_single': npz_data['rew_single'],
            'reg_coop': npz_data['reg_coop'],
            'rew_coop': npz_data['rew_coop']
        }

    if not data_all:
        print("Error: No data found. Check your suffix.")
        return

    window_size = min(50000, len(t_steps) // 10)
    save_suffix = f"COMPARE_6Lines_Single_vs_Coop_{args.suffix}"
    if args.zoom is not None: save_suffix += f"_zoom{int(args.zoom)}"
    os.makedirs(results_dir, exist_ok=True)

    # PLOT 1: Cumulative Regret
    plt.figure(figsize=(10, 6), dpi=150)
    handles_single, labels_single, handles_coop, labels_coop = [], [], [], []
    for pol, d in data_all.items():
        hs, = plt.plot(t_steps, d['reg_single'], color=colors[pol], linestyle='--', linewidth=1.5)
        handles_single.append(hs); labels_single.append(f"{pol} (Single-Agent)")
        hc, = plt.plot(t_steps, d['reg_coop'], color=colors[pol], linestyle='-', linewidth=2.0)
        handles_coop.append(hc); labels_coop.append(f"{pol} (Coop 25 Agents)")
        
    plt.title(f"Cumulative Regret: Single-Agent vs Cooperative Network | K={k_arms}")
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Cumulative Regret")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ordered_handles = handles_single + handles_coop
    ordered_labels = labels_single + labels_coop
    plt.legend(ordered_handles, ordered_labels, loc="upper left", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_1_regret_{save_suffix}.png"))

    print(f"\n6-Line Comparison Plots saved to results/ with suffix '{save_suffix}'")
    if not args.no_show: plt.show()

if __name__ == "__main__":
    main()