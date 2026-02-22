from netpyne import specs

try:
    from __main__ import cfg
except:
    from tut8_cfg import cfg

netParams = specs.NetParams()

# --- single-compartment HH pyramidal cell ---
PYRcell = {'secs': {}}
PYRcell['secs']['soma'] = {
    'geom': {'diam': 18.8, 'L': 18.8, 'Ra': 123.0},
    'mechs': {
        'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}
    },
}
netParams.cellParams['PYR'] = PYRcell

# --- two populations: S (sensory) and M (motor), 20 cells each ---
netParams.popParams['S'] = {'cellType': 'PYR', 'numCells': 20}
netParams.popParams['M'] = {'cellType': 'PYR', 'numCells': 20}

# --- excitatory synapse whose tau2 is controlled by cfg ---
netParams.synMechParams['exc'] = {
    'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': cfg.synMechTau2, 'e': 0,
}

# --- background poisson drive ---
netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 10, 'noise': 0.5}
netParams.stimTargetParams['bkg->PYR'] = {
    'source': 'bkg', 'conds': {'cellType': 'PYR'},
    'weight': 0.01, 'delay': 5, 'synMech': 'exc',
}

# --- S -> M connectivity, weight controlled by cfg ---
netParams.connParams['S->M'] = {
    'preConds': {'pop': 'S'},
    'postConds': {'pop': 'M'},
    'probability': 0.5,
    'weight': cfg.connWeight,
    'delay': 5,
    'synMech': 'exc',
}
