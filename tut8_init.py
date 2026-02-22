from netpyne import sim

simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault='tut8_cfg.py',
    netParamsDefault='tut8_netParams.py',
)

sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)
