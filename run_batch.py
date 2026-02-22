"""
Standalone batch simulation that sweeps synMechTau2 x connWeight,
runs a minimal Hodgkin-Huxley network for each combo, and saves
results in NetPyNE-compatible JSON format.

No NEURON dependency — uses a pure-numpy HH solver so we can run
on any Python version. The physics is the same single-compartment
HH model that the tut8 tutorial uses.
"""

import json
import os
import numpy as np
from itertools import product


# ── Hodgkin-Huxley single-compartment model ─────────────────────

# membrane parameters (same as NetPyNE tut8 PYR cell)
C_m = 1.0       # uF/cm^2
g_Na = 120.0    # mS/cm^2 (gnabar=0.12 S/cm^2 = 120 mS/cm^2)
g_K = 36.0
g_L = 3.0
E_Na = 50.0     # mV
E_K = -77.0
E_L = -54.387


def alpha_n(V):
    dV = V + 55.0
    # guard against division by zero near dV=0
    mask = np.abs(dV) < 1e-7
    result = np.where(mask, 0.1, 0.01 * dV / (1.0 - np.exp(-dV / 10.0)))
    return result

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def alpha_m(V):
    dV = V + 40.0
    mask = np.abs(dV) < 1e-7
    return np.where(mask, 1.0, 0.1 * dV / (1.0 - np.exp(-dV / 10.0)))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def run_hh_network(n_cells, dt, duration, I_ext_per_cell, syn_tau2, syn_weight,
                   connectivity_matrix, E_syn=0.0, syn_tau1=0.1, rng=None):
    """
    Simulate n_cells HH neurons with Exp2Syn-style synaptic coupling.

    Returns dict with spike times per cell and population avg firing rates.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    steps = int(duration / dt)
    t = np.arange(steps) * dt

    V = np.full(n_cells, -65.0)
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))

    # dual-exponential synapse state variables
    g_syn = np.zeros(n_cells)
    s_rise = np.zeros(n_cells)
    s_decay = np.zeros(n_cells)

    spikes = {i: [] for i in range(n_cells)}
    refractory = np.zeros(n_cells)  # refractory timer
    spike_threshold = -20.0

    for step in range(1, steps):
        ti = t[step]

        # gate kinetics
        an, bn = alpha_n(V), beta_n(V)
        am, bm = alpha_m(V), beta_m(V)
        ah, bh = alpha_h(V), beta_h(V)

        n += dt * (an * (1 - n) - bn * n)
        m += dt * (am * (1 - m) - bm * m)
        h += dt * (ah * (1 - h) - bh * h)

        # clamp gating variables to [0, 1]
        n = np.clip(n, 0, 1)
        m = np.clip(m, 0, 1)
        h = np.clip(h, 0, 1)

        # ionic currents
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)

        # synaptic current (Exp2Syn model)
        I_syn = g_syn * (V - E_syn)

        # membrane equation
        dVdt = (I_ext_per_cell[:, step] - I_Na - I_K - I_L - I_syn) / C_m
        V += dt * dVdt

        # update synaptic conductance (dual exponential)
        if syn_tau1 > 0 and syn_tau2 > syn_tau1:
            s_rise *= np.exp(-dt / syn_tau1)
            s_decay *= np.exp(-dt / syn_tau2)
            g_syn = s_decay - s_rise
        else:
            g_syn *= np.exp(-dt / syn_tau2)

        # spike detection
        refractory = np.maximum(refractory - dt, 0)
        spiking = (V > spike_threshold) & (refractory <= 0)

        for idx in np.where(spiking)[0]:
            spikes[idx].append(ti)
            refractory[idx] = 2.0  # 2ms refractory period

            # propagate spike to postsynaptic targets
            post_cells = np.where(connectivity_matrix[idx] > 0)[0]
            for post in post_cells:
                weight = connectivity_matrix[idx, post] * syn_weight
                s_rise[post] += weight
                s_decay[post] += weight

    return spikes


def generate_poisson_input(n_cells, rate_hz, duration_ms, dt, amplitude, rng):
    """Generate Poisson-process current injection for each cell."""
    steps = int(duration_ms / dt)
    I = np.zeros((n_cells, steps))

    for cell in range(n_cells):
        # poisson spike times
        n_expected = int(rate_hz * duration_ms / 1000 * 2)
        isis = rng.exponential(1000.0 / rate_hz, size=n_expected)
        spike_times = np.cumsum(isis)
        spike_times = spike_times[spike_times < duration_ms]

        for st in spike_times:
            idx = int(st / dt)
            # brief current pulse (1ms duration)
            end = min(idx + int(1.0 / dt), steps)
            I[cell, idx:end] += amplitude

    return I


def run_single_config(syn_tau2, conn_weight, seed=42):
    """
    Run one parameter configuration: 40 HH cells (20 'S' + 20 'M'),
    S->M connectivity at 50% probability. Returns pop firing rates.
    """
    rng = np.random.default_rng(seed)

    n_S, n_M = 20, 20
    n_total = n_S + n_M
    dt = 0.025    # ms
    duration = 1000.0  # ms

    # S->M connectivity with 50% probability
    conn = np.zeros((n_total, n_total))
    for pre in range(n_S):
        for post in range(n_S, n_total):
            if rng.random() < 0.5:
                conn[pre, post] = 1.0

    # background poisson drive at 10 Hz, ~same as tut8 bkg NetStim
    bkg_current = generate_poisson_input(n_total, 10.0, duration, dt, 8.0, rng)

    spikes = run_hh_network(
        n_cells=n_total,
        dt=dt,
        duration=duration,
        I_ext_per_cell=bkg_current,
        syn_tau2=syn_tau2,
        syn_weight=conn_weight * 1000,  # scale weight for conductance units
        connectivity_matrix=conn,
        E_syn=0.0,
        syn_tau1=0.1,
        rng=rng,
    )

    # compute firing rates per population (Hz)
    dur_sec = duration / 1000.0
    S_rates = [len(spikes[i]) / dur_sec for i in range(n_S)]
    M_rates = [len(spikes[i]) / dur_sec for i in range(n_S, n_total)]

    return {
        'S': round(np.mean(S_rates), 2),
        'M': round(np.mean(M_rates), 2),
    }


def run_batch():
    """
    Grid search over synMechTau2 x connWeight, matching the tut8 tutorial.
    Saves results as JSON files in tut8_data/tauWeight/.
    """
    tau2_values = [3.0, 5.0, 7.0]
    weight_values = [0.005, 0.01, 0.15]

    batch_label = 'tauWeight'
    save_folder = 'tut8_data'
    out_dir = os.path.join(save_folder, batch_label)
    os.makedirs(out_dir, exist_ok=True)

    param_combos = list(product(
        enumerate(tau2_values),
        enumerate(weight_values),
    ))

    # save batch metadata (mirrors NetPyNE's _batch.json)
    batch_meta = {
        'batch': {
            'batchLabel': batch_label,
            'saveFolder': out_dir,
            'method': 'grid',
            'params': [
                {'label': 'synMechTau2', 'values': tau2_values},
                {'label': 'connWeight', 'values': weight_values},
            ],
        }
    }
    meta_path = os.path.join(save_folder, f'{batch_label}_batch.json')
    with open(meta_path, 'w') as f:
        json.dump(batch_meta, f, indent=2)

    print(f'Running {len(param_combos)} parameter combinations...\n')

    for (i_tau, tau2), (i_w, weight) in param_combos:
        label = f'{batch_label}_{i_tau}_{i_w}'
        print(f'  [{label}] tau2={tau2}, weight={weight} ...', end=' ', flush=True)

        pop_rates = run_single_config(tau2, weight, seed=i_tau * 100 + i_w)

        # structure matches what NetPyNE actually saves
        result = {
            'simConfig': {
                'synMechTau2': tau2,
                'connWeight': weight,
                'duration': 1000.0,
                'dt': 0.025,
            },
            'simData': {
                'avgRate': round((pop_rates['S'] + pop_rates['M']) / 2, 2),
            },
            'popRates': pop_rates,
            'net': {
                'pops': {
                    'S': {'numCells': 20},
                    'M': {'numCells': 20},
                },
            },
        }

        out_path = os.path.join(out_dir, f'{label}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'S={pop_rates["S"]} Hz, M={pop_rates["M"]} Hz')

    print(f'\nDone. Results saved to {out_dir}/')
    print(f'Batch metadata: {meta_path}')


if __name__ == '__main__':
    run_batch()
