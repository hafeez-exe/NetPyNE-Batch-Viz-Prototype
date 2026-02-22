from netpyne import specs
from netpyne.batch import Batch


def batch_tau_weight():
    params = specs.ODict()
    params['synMechTau2'] = [3.0, 5.0, 7.0]
    params['connWeight'] = [0.005, 0.01, 0.15]

    b = Batch(
        params=params,
        cfgFile='tut8_cfg.py',
        netParamsFile='tut8_netParams.py',
    )

    b.batchLabel = 'tauWeight'
    b.saveFolder = 'tut8_data'
    b.method = 'grid'
    b.runCfg = {
        'type': 'mpi_bulletin',
        'script': 'tut8_init.py',
        'skip': True,
    }

    b.run()


if __name__ == '__main__':
    batch_tau_weight()
