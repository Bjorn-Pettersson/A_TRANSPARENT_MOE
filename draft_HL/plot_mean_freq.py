import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_name = '/home/lee/T-MOE/test2.csv'

block_size = 20
start_index = 2

output_plot_file = 'mean_freq.png'

columns_to_analyze = [
    'Step',
    'sequence-moe-plt2 - layer0/expert0',
    'sequence-moe-plt2 - layer0/expert1',
    'sequence-moe-plt2 - layer0/expert2',
    'sequence-moe-plt2 - layer0/expert3'
]

expert_cols = [col for col in columns_to_analyze if col != 'Step']

df = pd.read_csv(file_name, usecols=columns_to_analyze)

df_filtered = df.iloc[start_index:].reset_index(drop=True)

group_key = df_filtered.index // block_size

df_block_mean = df_filtered.groupby(group_key)[expert_cols].mean()

block_start_step_index = df_filtered.groupby(group_key).apply(lambda x: x.index[0])
block_start_steps = df_filtered.loc[block_start_step_index, 'Step']

df_block_mean['block_start_step'] = block_start_steps.values
df_block_mean = df_block_mean.reset_index(drop=True)

print(df_block_mean[['block_start_step'] + expert_cols].head())

plt.figure(figsize=(12, 6))

for col in expert_cols:
    plt.plot(df_block_mean['block_start_step'], df_block_mean[col], label=col, marker='o', markersize=4, linewidth=1.5)

plt.title(f'Layer 0 Expert Frequencies ({block_size}-Step Block Mean from Index {start_index})', fontsize=14)
plt.xlabel(f'Training Step (Block Start Step corresponding to Index {start_index}, {start_index + block_size}, ...)', fontsize=12)
plt.ylabel('Block Mean Frequency', fontsize=12)
plt.legend(title='Expert', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig(output_plot_file)
plt.close()

print(f"saved: {output_plot_file}")
    
