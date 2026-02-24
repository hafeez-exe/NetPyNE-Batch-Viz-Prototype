import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
from itertools import product

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class HHParams:
    """Hodgkin-Huxley single-compartment parameters.

    Attributes:
        C_m: Membrane capacitance (uF/cm^2).
        g_Na: Maximum sodium conductance (mS/cm^2).
        g_K: Maximum potassium conductance (mS/cm^2).
        g_L: Maximum leak conductance (mS/cm^2).
        E_Na: Sodium reversal potential (mV).
        E_K: Potassium reversal potential (mV).
        E_L: Leak reversal potential (mV).
    """
    C_m: float = 1.0
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 3.0
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387


@dataclass
class SimulationConfig:
    """Configuration for a single network simulation run.

    Attributes:
        duration_ms: Total duration of the simulation in milliseconds.
        dt: Integration time step in milliseconds.
        n_s: Number of sensory (S) cells.
        n_m: Number of motor (M) cells.
        connection_prob: Probability of connection from S to M cells.
        bkg_rate_hz: Background Poisson input rate in Hz.
        bkg_amplitude: Background Poisson input amplitude.
        spike_threshold: Membrane potential threshold for spike detection (mV).
    """
    duration_ms: float = 1000.0
    dt: float = 0.025
    n_s: int = 20
    n_m: int = 20
    connection_prob: float = 0.5
    bkg_rate_hz: float = 10.0
    bkg_amplitude: float = 8.0
    spike_threshold: float = -20.0


@dataclass
class BatchConfig:
    """Configuration for the batch sweep.

    Attributes:
        tau2_values: List of synaptic decay time constants to explore.
        weight_values: List of synaptic weights to explore.
        batch_label: Label for the batch run.
        save_folder: Output directory name.
    """
    tau2_values: List[float]
    weight_values: List[float]
    batch_label: str = "tauWeight"
    save_folder: str = "tut8_data"


def alpha_n(v_m: np.ndarray) -> np.ndarray:
    """Calculate alpha_n rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    dv = v_m + 55.0
    mask = np.abs(dv) < 1e-7
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(mask, 0.1, 0.01 * dv / (1.0 - np.exp(-dv / 10.0)))
    return result


def beta_n(v_m: np.ndarray) -> np.ndarray:
    """Calculate beta_n rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    return 0.125 * np.exp(-(v_m + 65.0) / 80.0)


def alpha_m(v_m: np.ndarray) -> np.ndarray:
    """Calculate alpha_m rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    dv = v_m + 40.0
    mask = np.abs(dv) < 1e-7
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(mask, 1.0, 0.1 * dv / (1.0 - np.exp(-dv / 10.0)))
    return result


def beta_m(v_m: np.ndarray) -> np.ndarray:
    """Calculate beta_m rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    return 4.0 * np.exp(-(v_m + 65.0) / 18.0)


def alpha_h(v_m: np.ndarray) -> np.ndarray:
    """Calculate alpha_h rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    return 0.07 * np.exp(-(v_m + 65.0) / 20.0)


def beta_h(v_m: np.ndarray) -> np.ndarray:
    """Calculate beta_h rate constant for the Hodgkin-Huxley system.

    Args:
        v_m: Array of membrane potentials.

    Returns:
        Array of rate constants.
    """
    return 1.0 / (1.0 + np.exp(-(v_m + 35.0) / 10.0))


def run_hh_network(
    n_cells: int,
    dt: float,
    duration: float,
    i_ext_per_cell: np.ndarray,
    syn_tau2: float,
    syn_weight: float,
    connectivity_matrix: np.ndarray,
    e_syn: float = 0.0,
    syn_tau1: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    params: Optional[HHParams] = None,
    spike_threshold: float = -20.0
) -> Dict[int, List[float]]:
    """Simulate a network of Hodgkin-Huxley neurons with Exp2Syn coupling.

    Args:
        n_cells: Number of cells in the network.
        dt: Integration time step in ms.
        duration: Total duration of simulation in ms.
        i_ext_per_cell: 2D array of external current injection [cell, time].
        syn_tau2: Synaptic decay time constant.
        syn_weight: Synaptic weight.
        connectivity_matrix: 2D array representing synaptic connectivity.
        e_syn: Synaptic reversal potential.
        syn_tau1: Synaptic rise time constant.
        rng: NumPy random generator instance.
        params: Hodgkin-Huxley parameters dataclass instance.
        spike_threshold: Membrane potential required to register a spike.

    Returns:
        Dictionary mapping cell index to a list of spike times.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if params is None:
        params = HHParams()

    steps = int(duration / dt)
    t = np.arange(steps) * dt

    v_m = np.full(n_cells, -65.0)
    n = alpha_n(v_m) / (alpha_n(v_m) + beta_n(v_m))
    m = alpha_m(v_m) / (alpha_m(v_m) + beta_m(v_m))
    h = alpha_h(v_m) / (alpha_h(v_m) + beta_h(v_m))

    g_syn = np.zeros(n_cells)
    s_rise = np.zeros(n_cells)
    s_decay = np.zeros(n_cells)

    spikes: Dict[int, List[float]] = {i: [] for i in range(n_cells)}
    refractory = np.zeros(n_cells)

    for step in range(1, steps):
        ti = t[step]

        an, bn = alpha_n(v_m), beta_n(v_m)
        am, bm = alpha_m(v_m), beta_m(v_m)
        ah, bh = alpha_h(v_m), beta_h(v_m)

        n += dt * (an * (1.0 - n) - bn * n)
        m += dt * (am * (1.0 - m) - bm * m)
        h += dt * (ah * (1.0 - h) - bh * h)

        n = np.clip(n, 0.0, 1.0)
        m = np.clip(m, 0.0, 1.0)
        h = np.clip(h, 0.0, 1.0)

        i_na = params.g_Na * (m**3) * h * (v_m - params.E_Na)
        i_k = params.g_K * (n**4) * (v_m - params.E_K)
        i_l = params.g_L * (v_m - params.E_L)

        i_syn = g_syn * (v_m - e_syn)

        dvdt = (i_ext_per_cell[:, step] - i_na - i_k - i_l - i_syn) / params.C_m
        v_m += dt * dvdt

        if syn_tau1 > 0 and syn_tau2 > syn_tau1:
            s_rise *= np.exp(-dt / syn_tau1)
            s_decay *= np.exp(-dt / syn_tau2)
            g_syn = s_decay - s_rise
        else:
            g_syn *= np.exp(-dt / syn_tau2)

        refractory = np.maximum(refractory - dt, 0.0)
        spiking = (v_m > spike_threshold) & (refractory <= 0.0)

        for idx in np.where(spiking)[0]:
            spikes[idx].append(ti)
            refractory[idx] = 2.0

            post_cells = np.where(connectivity_matrix[idx] > 0)[0]
            for post in post_cells:
                weight = connectivity_matrix[idx, post] * syn_weight
                s_rise[post] += weight
                s_decay[post] += weight

    return spikes


def generate_poisson_input(
    n_cells: int,
    rate_hz: float,
    duration_ms: float,
    dt: float,
    amplitude: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Generate Poisson-process current injection for each cell.

    Args:
        n_cells: Number of cells receiving input.
        rate_hz: Mean firing rate of the Poisson process.
        duration_ms: Total duration of the simulation in ms.
        dt: Integration time step in ms.
        amplitude: Amplitude of the current pulse injected upon an event.
        rng: NumPy random generator instance.

    Returns:
        2D array of external currents [cell, time].
    """
    steps = int(duration_ms / dt)
    i_ext = np.zeros((n_cells, steps))

    for cell in range(n_cells):
        n_expected = int(rate_hz * duration_ms / 1000.0 * 2.0)
        isis = rng.exponential(1000.0 / rate_hz, size=n_expected)
        spike_times = np.cumsum(isis)
        spike_times = spike_times[spike_times < duration_ms]

        for st in spike_times:
            idx = int(st / dt)
            end = min(idx + int(1.0 / dt), steps)
            i_ext[cell, idx:end] += amplitude

    return i_ext


def run_single_config(
    syn_tau2: float,
    conn_weight: float,
    seed: int = 42,
    config: Optional[SimulationConfig] = None
) -> Dict[str, float]:
    """Execute one parameter configuration for the population.

    Args:
        syn_tau2: Synaptic decay time constant.
        conn_weight: Connection weight scalar.
        seed: Random seed for reproducibility.
        config: SimulationConfig dataclass instance.

    Returns:
        Dictionary containing the mean firing rate (Hz) for S and M populations.
    """
    if config is None:
        config = SimulationConfig()

    rng = np.random.default_rng(seed)
    n_total = config.n_s + config.n_m

    conn = np.zeros((n_total, n_total))
    for pre in range(config.n_s):
        for post in range(config.n_s, n_total):
            if rng.random() < config.connection_prob:
                conn[pre, post] = 1.0

    bkg_current = generate_poisson_input(
        n_cells=n_total,
        rate_hz=config.bkg_rate_hz,
        duration_ms=config.duration_ms,
        dt=config.dt,
        amplitude=config.bkg_amplitude,
        rng=rng
    )

    spikes = run_hh_network(
        n_cells=n_total,
        dt=config.dt,
        duration=config.duration_ms,
        i_ext_per_cell=bkg_current,
        syn_tau2=syn_tau2,
        syn_weight=conn_weight * 1000.0,
        connectivity_matrix=conn,
        rng=rng,
        spike_threshold=config.spike_threshold
    )

    dur_sec = config.duration_ms / 1000.0
    s_rates = [len(spikes[i]) / dur_sec for i in range(config.n_s)]
    m_rates = [len(spikes[i]) / dur_sec for i in range(config.n_s, n_total)]

    return {
        'S': round(float(np.mean(s_rates)), 2),
        'M': round(float(np.mean(m_rates)), 2),
    }


def run_batch(config: Optional[BatchConfig] = None) -> None:
    """Execute grid search parameter sweep and export data to JSON limits.

    Args:
        config: BatchConfig dataclass instance containing sweep parameters.
    """
    if config is None:
        config = BatchConfig(
            tau2_values=[3.0, 5.0, 7.0],
            weight_values=[0.005, 0.01, 0.15]
        )

    sim_cfg = SimulationConfig()
    out_dir = os.path.join(config.save_folder, config.batch_label)
    os.makedirs(out_dir, exist_ok=True)

    param_combos = list(product(
        enumerate(config.tau2_values),
        enumerate(config.weight_values)
    ))

    batch_meta = {
        'batch': {
            'batchLabel': config.batch_label,
            'saveFolder': out_dir,
            'method': 'grid',
            'params': [
                {'label': 'synMechTau2', 'values': config.tau2_values},
                {'label': 'connWeight', 'values': config.weight_values},
            ],
        }
    }

    meta_path = os.path.join(config.save_folder, f'{config.batch_label}_batch.json')
    with open(meta_path, 'w') as f:
        json.dump(batch_meta, f, indent=2)

    for (i_tau, tau2), (i_w, weight) in param_combos:
        label = f'{config.batch_label}_{i_tau}_{i_w}'
        
        pop_rates = run_single_config(
            syn_tau2=tau2,
            conn_weight=weight,
            seed=i_tau * 100 + i_w,
            config=sim_cfg
        )

        result = {
            'simConfig': {
                'synMechTau2': tau2,
                'connWeight': weight,
                'duration': sim_cfg.duration_ms,
                'dt': sim_cfg.dt,
            },
            'simData': {
                'avgRate': round((pop_rates['S'] + pop_rates['M']) / 2.0, 2),
            },
            'popRates': pop_rates,
            'net': {
                'pops': {
                    'S': {'numCells': sim_cfg.n_s},
                    'M': {'numCells': sim_cfg.n_m},
                },
            },
        }

        out_path = os.path.join(out_dir, f'{label}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    run_batch()
