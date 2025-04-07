import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import argparse

def parse_tensor(tensor_str):
    pattern = r'tensor\(\[(.*?)\]'
    match = re.search(pattern, tensor_str, re.DOTALL)
    
    if match:
        numbers_str = match.group(1)
        numbers_str = re.sub(r'\s+', ' ', numbers_str)
        values = [float(x.strip()) for x in numbers_str.split(',') if x.strip()]
        return values
    else:
        all_numbers = re.findall(r'[-+]?\d*\.\d+|\d+', tensor_str)
        if all_numbers:
            return [float(x) for x in all_numbers]
        return []

def process_data(csv_path):
    df = pd.read_csv(csv_path)
    q_values_avg = []
    
    for q_str in df['q_value']:
        values = parse_tensor(q_str)
        if values:
            q_values_avg.append(np.mean(values))
        else:
            q_values_avg.append(np.nan)
    
    df['avg_q_value'] = q_values_avg
    
    print(f"Processed {len(df)} rows of data")
    print(f"Successfully extracted Q-values for {sum(~np.isnan(q_values_avg))} rows")
    
    if sum(~np.isnan(q_values_avg)) > 0:
        first_valid_idx = next((i for i, x in enumerate(q_values_avg) if not np.isnan(x)), None)
        if first_valid_idx is not None:
            print(f"Example - Row {first_valid_idx}")
            print(f"Original tensor: {df['q_value'].iloc[first_valid_idx][:100]}...")
            print(f"Extracted values (first 5): {parse_tensor(df['q_value'].iloc[first_valid_idx])[:5]}")
            print(f"Average Q-value: {q_values_avg[first_valid_idx]}")
    
    return df

def plot_rewards_avg(df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['avg_reward_last_100'], marker='o', linestyle='-', color='blue')
    plt.title('Average Reward (Last 100 Episodes) vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'avg_reward_vs_episode.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reward plot to: {output_path}")

def plot_rewards(df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['reward'], marker='o', linestyle='-', color='blue')
    plt.title('Average Reward (Last 100 Episodes) vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Last 100 Episodes)')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'reward_vs_episode.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reward plot to: {output_path}")


def plot_q_values(df, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['avg_q_value'], marker='o', linestyle='-', color='green')
    plt.title('Average Q-Value vs Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'avg_q_value_vs_episode.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Q-value plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate RL training plots from CSV data')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(os.path.abspath(args.csv_path))
    
    df = process_data(args.csv_path)
    
    plot_rewards(df, output_dir)
    plot_rewards_avg(df, output_dir)
    plot_q_values(df, output_dir)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main()