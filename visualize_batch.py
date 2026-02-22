"""
Reads the batch simulation output from tut8_data/tauWeight/ and
generates a heatmap showing how synMechTau2 and connWeight affect
the M-population firing rate.

Usage:
    python visualize_batch.py
"""

import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from collections import OrderedDict


def load_batch_results(data_folder='tut8_data', batch_label='tauWeight'):
    """
    Read the batch metadata and all individual simulation JSONs.
    Returns a DataFrame with one row per parameter combination.
    """
    meta_path = os.path.join(data_folder, f'{batch_label}_batch.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)['batch']

    params = meta['params']
    param_labels = [p['label'] for p in params]
    param_values = [p['values'] for p in params]

    rows = []
    for indices in product(*[range(len(v)) for v in param_values]):
        combo_str = '_'.join(str(i) for i in indices)
        sim_label = f'{batch_label}_{combo_str}'
        sim_path = os.path.join(data_folder, batch_label, f'{sim_label}.json')

        try:
            with open(sim_path, 'r') as f:
                result = json.load(f, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            print(f'  warning: {sim_path} not found, skipping')
            continue

        row = {}
        for label, vals, idx in zip(param_labels, param_values, indices):
            row[label] = vals[idx]

        # grab population firing rates
        pop_rates = result.get('popRates', {})
        for pop_name, rate in pop_rates.items():
            row[f'{pop_name}_rate'] = rate

        row['avg_rate'] = result.get('simData', {}).get('avgRate', np.nan)
        row['sim_label'] = sim_label
        rows.append(row)

    return pd.DataFrame(rows)


def plot_firing_rate_heatmap(df, save_path='figures/fitness_heatmap.png'):
    """
    Create a heatmap: synMechTau2 (y) vs connWeight (x), color = M-pop rate.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pivot = df.pivot_table(
        values='M_rate',
        index='synMechTau2',
        columns='connWeight',
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        linewidths=1,
        linecolor='#333',
        square=True,
        cbar_kws={'label': 'Firing Rate (Hz)', 'shrink': 0.8},
        ax=ax,
    )

    ax.set_title(
        'M-Population Firing Rate\n'
        'Grid Search: synMechTau2 × connWeight',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.set_xlabel('Connection Weight (connWeight)', fontsize=12)
    ax.set_ylabel('Synaptic Decay Time (synMechTau2, ms)', fontsize=12)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved heatmap → {save_path}')
    plt.close()


def plot_dual_population(df, save_path='figures/dual_pop_heatmap.png'):
    """
    Side-by-side heatmaps for S and M populations.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, pop, title in zip(axes, ['S_rate', 'M_rate'], ['S (Sensory)', 'M (Motor)']):
        pivot = df.pivot_table(values=pop, index='synMechTau2', columns='connWeight')

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='viridis',
            linewidths=0.8,
            linecolor='#444',
            square=True,
            cbar_kws={'label': 'Hz', 'shrink': 0.75},
            ax=ax,
        )
        ax.set_title(f'{title} Population', fontsize=13, fontweight='bold')
        ax.set_xlabel('connWeight', fontsize=11)
        ax.set_ylabel('synMechTau2 (ms)', fontsize=11)

    fig.suptitle(
        'Batch Fitness Report: Population Firing Rates Across Parameter Sweep',
        fontsize=14, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved dual heatmap → {save_path}')
    plt.close()


def plot_rate_difference(df, save_path='figures/rate_diff_heatmap.png'):
    """
    Heatmap of M_rate - S_rate, showing how much the downstream
    population is driven by each parameter combination.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df['rate_diff'] = df['M_rate'] - df['S_rate']

    pivot = df.pivot_table(values='rate_diff', index='synMechTau2', columns='connWeight')

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=1,
        linecolor='#333',
        square=True,
        cbar_kws={'label': 'ΔRate: M − S (Hz)', 'shrink': 0.8},
        ax=ax,
    )

    ax.set_title(
        'Synaptic Drive Effect\n'
        'M-population minus S-population firing rate',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.set_xlabel('connWeight', fontsize=12)
    ax.set_ylabel('synMechTau2 (ms)', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved rate difference heatmap → {save_path}')
    plt.close()


if __name__ == '__main__':
    print('Loading batch results...')
    df = load_batch_results()
    print(f'Loaded {len(df)} configurations\n')
    print(df[['synMechTau2', 'connWeight', 'S_rate', 'M_rate']].to_string(index=False))
    print()

    plot_firing_rate_heatmap(df)
    plot_dual_population(df)
    plot_rate_difference(df)

    print('\nAll figures saved to figures/')
