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
    parser = argparse.ArgumentParser(description="Replot MAB results dynamically.")
    parser.add_argument("--file", type=str, required=True, help="Specific .npz file to load")
    parser.add_argument("--single", action="store_true", help="Include Single-Agent Baseline")
    parser.add_argument("--coop", action="store_true", help="Include Fully Cooperative Multi-Agent")
    parser.add_argument("--malicious", action="store_true", help="Include Malicious Scenarios")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--zoom", type=float, nargs='?', const=80.0, default=None, help="Zoom-in percentage (e.g. 80)")
    args = parser.parse_args()

    if not (args.single or args.coop or args.malicious):
        args.single = True; args.coop = True; args.malicious = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    data_path = os.path.join(results_dir, args.file)
    if not os.path.exists(data_path):
        print(f"Error: '{data_path}' not found.")
        return

    npz_data = np.load(data_path, allow_pickle=True)
    data = {key: npz_data[key] for key in npz_data.files}

    t_steps = data['t_steps']
    k_arms = data.get('k_arms', 'Unknown')
    policy = data.get('policy', 'UCB')
    
    mal_colors = ['tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    mark_interval = max(1, int(len(t_steps) / 10))
    window_size = min(50000, len(t_steps) // 10)
    
    plot_flags = []
    if args.single: plot_flags.append("single")
    if args.coop: plot_flags.append("coop")
    if args.malicious: plot_flags.append("malicious")
    if args.zoom is not None: plot_flags.append(f"zoom{int(args.zoom)}")
    
    suffix = args.file.replace('.npz', '') + "_replot_" + "_".join(plot_flags)
    os.makedirs(results_dir, exist_ok=True)
    
    mal_keys = sorted([k for k in data.keys() if k.startswith('reg_mal_')], key=lambda x: int(x.split('_')[3].replace('m', '')))

    # PLOT 1: Regret
    plt.figure(figsize=(9, 5.5), dpi=150)
    if args.single and 'reg_single' in data:
        plt.plot(t_steps, data['reg_single'], label="Single-Agent Baseline", ls='--', color='tab:blue', marker='|', markevery=mark_interval, markersize=10)
    if args.coop and 'reg_coop' in data:
        plt.plot(t_steps, data['reg_coop'], label="Fully Coop. (0% Malicious)", ls=':', color='tab:red', marker='|', markevery=mark_interval, markersize=10)
    if args.malicious:
        for c_idx, key in enumerate(mal_keys):
            n_val = int(key.split('_')[2].replace('n', ''))
            m_val = int(key.split('_')[3].replace('m', ''))
            pct = int((m_val / (n_val + m_val)) * 100)
            plt.plot(t_steps, data[key], label=f"Malicious Attack ({pct}% Malicious)", ls='-.', color=mal_colors[c_idx % len(mal_colors)], marker='|', markevery=mark_interval, markersize=10)

    plt.title(f"Cumulative Regret (Total 25 Agents) | {policy} - K={k_arms}")
    plt.xlabel(r"Time Horizon ($T$)")
    plt.ylabel(r"Average Cumulative Regret ($\frac{1}{n} \sum_{i=1}^n R_T^{(i)}$)")
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_1_regret_{suffix}.png"))

    # PLOT 3: Rolling Mean
    plt.figure(figsize=(9, 5.5), dpi=150)
    rm_dict = {}
    if args.single and 'rew_single' in data:
        rm_dict['single'] = get_rolling_mean(data['rew_single'], window_size)
        plt.plot(t_steps, rm_dict['single'], label="Single-Agent Baseline", ls='--', color='tab:blue')
    if args.coop and 'rew_coop' in data:
        rm_dict['coop'] = get_rolling_mean(data['rew_coop'], window_size)
        plt.plot(t_steps, rm_dict['coop'], label="Fully Coop. (0% Malicious)", ls=':', color='tab:red')
    if args.malicious:
        for c_idx, key in enumerate(mal_keys):
            rew_key = key.replace('reg_', 'rew_')
            n_val = int(key.split('_')[2].replace('n', ''))
            m_val = int(key.split('_')[3].replace('m', ''))
            pct = int((m_val / (n_val + m_val)) * 100)
            rm_dict[key] = get_rolling_mean(data[rew_key], window_size)
            plt.plot(t_steps, rm_dict[key], label=f"Malicious Attack ({pct}% Malicious)", ls='-.', color=mal_colors[c_idx % len(mal_colors)])

    plt.title(f"{window_size}-Round Rolling Mean (Total 25 Agents) | {policy} - K={k_arms}")
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
    plt.savefig(os.path.join(results_dir, f"plot_3_rolling_reward_{suffix}.png"))

    print(f"Replots successfully generated to results/ with suffix: '{suffix}'")
    if not args.no_show: plt.show()
    else: plt.close('all')

if __name__ == "__main__":
    main()