import json
import os
from dataclasses import dataclass
from itertools import product
from collections import OrderedDict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class VisualizationConfig:
    """Configuration for batch visualization data loading and export.

    Attributes:
        data_folder: Root directory containing batch JSONs.
        batch_label: Label forming the prefix of the batch folder and JSON files.
        figures_dir: Output directory for saving heatmaps.
    """
    data_folder: str = 'tut8_data'
    batch_label: str = 'tauWeight'
    figures_dir: str = 'figures'


def load_batch_results(config: Optional[VisualizationConfig] = None) -> pd.DataFrame:
    """Read batch metadata and parse simulation JSON payloads into a DataFrame.

    Args:
        config: Data loading configuration settings.

    Returns:
        A DataFrame structure containing parameter variables and resulting firing rates.
    """
    if config is None:
        config = VisualizationConfig()

    meta_path = os.path.join(config.data_folder, f'{config.batch_label}_batch.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)['batch']

    params = meta.get('params', [])
    param_labels = [p['label'] for p in params]
    param_values = [p['values'] for p in params]

    rows: List[Dict[str, Any]] = []

    for indices in product(*[range(len(v)) for v in param_values]):
        combo_str = '_'.join(str(i) for i in indices)
        sim_label = f'{config.batch_label}_{combo_str}'
        sim_path = os.path.join(config.data_folder, config.batch_label, f'{sim_label}.json')

        try:
            with open(sim_path, 'r') as f:
                result = json.load(f, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            continue

        row: Dict[str, Any] = {}
        for label, vals, idx in zip(param_labels, param_values, indices):
            row[label] = vals[idx]

        pop_rates = result.get('popRates', {})
        for pop_name, rate in pop_rates.items():
            row[f'{pop_name}_rate'] = rate

        row['avg_rate'] = result.get('simData', {}).get('avgRate', np.nan)
        row['sim_label'] = sim_label
        rows.append(row)

    return pd.DataFrame(rows)


def plot_firing_rate_heatmap(
    df: pd.DataFrame,
    save_path: str = "figures/fitness_heatmap.png"
) -> None:
    """Render and save a heatmap showing M-population rates mapped to scanned parameters.

    Args:
        df: Merged output DataFrame linking parameters to recorded rates.
        save_path: Target generic output graph filepath.
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
    plt.close(fig)


def plot_dual_population(
    df: pd.DataFrame,
    save_path: str = "figures/dual_pop_heatmap.png"
) -> None:
    """Render side-by-side comparative heatmaps displaying distinct S vs M population data.

    Args:
        df: Merged output DataFrame handling experimental recordings.
        save_path: Output graph destination string.
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
    plt.close(fig)


def plot_rate_difference(
    df: pd.DataFrame,
    save_path: str = "figures/rate_diff_heatmap.png"
) -> None:
    """Compute and distribute subtracted rate matrix indicating driver impact (M minus S).

    Args:
        df: Processed recording sets holding populations.
        save_path: Relative string destination defining output folder location.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_copy = df.copy()
    df_copy['rate_diff'] = df_copy['M_rate'] - df_copy['S_rate']

    pivot = df_copy.pivot_table(values='rate_diff', index='synMechTau2', columns='connWeight')

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
    plt.close(fig)


if __name__ == '__main__':
    config = VisualizationConfig()
    df_results = load_batch_results(config)
    csv_path = os.path.join(config.data_folder, "fitness_table.csv")
    df_results.to_csv(csv_path, index=False)

    plot_firing_rate_heatmap(df_results)
    plot_dual_population(df_results)
    plot_rate_difference(df_results)
