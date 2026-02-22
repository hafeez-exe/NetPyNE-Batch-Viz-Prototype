from netpyne import specs

cfg = specs.SimConfig()

cfg.duration = 1 * 1e3          # 1 second
cfg.dt = 0.025
cfg.verbose = False
cfg.recordTraces = {'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}}
cfg.recordStep = 0.1
cfg.filename = 'tut8'
cfg.saveJson = True
cfg.printPopAvgRates = True

cfg.analysis['plotRaster'] = {'saveFig': True}
cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True}

cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']

# these two get swept by the batch script
cfg.synMechTau2 = 5
cfg.connWeight = 0.01
