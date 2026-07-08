import os
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Export NPZ to CSV.")
    parser.add_argument("--file", type=str, required=True, help="Filename of the .npz file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    
    npz_path = os.path.join(results_dir, args.file)
    csv_filename = args.file.replace('.npz', '.csv')
    csv_path = os.path.join(results_dir, csv_filename)
    
    if not os.path.exists(npz_path):
        print(f"Error: Could not find '{npz_path}'.")
        return

    data = np.load(npz_path, allow_pickle=True)
    csv_data = {"Time_Step": data['t_steps']}
    
    if 'reg_single' in data.files: csv_data['Regret_Single_Agent'] = data['reg_single']
    if 'reg_coop' in data.files: csv_data['Regret_Coop_0pct'] = data['reg_coop']
    if 'rew_single' in data.files: csv_data['Reward_Single_Agent'] = data['rew_single']
    if 'rew_coop' in data.files: csv_data['Reward_Coop_0pct'] = data['rew_coop']

    for key in data.files:
        if key.startswith('reg_mal_'):
            n_val = int(key.split('_')[2].replace('n', ''))
            m_val = int(key.split('_')[3].replace('m', ''))
            pct = int((m_val / (n_val + m_val)) * 100)
            csv_data[f'Regret_Malicious_{pct}pct'] = data[key]
        elif key.startswith('rew_mal_'):
            n_val = int(key.split('_')[2].replace('n', ''))
            m_val = int(key.split('_')[3].replace('m', ''))
            pct = int((m_val / (n_val + m_val)) * 100)
            csv_data[f'Reward_Malicious_{pct}pct'] = data[key]

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"Successfully exported to '{csv_filename}'!")

if __name__ == "__main__":
    main()