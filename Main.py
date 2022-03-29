import numpy as np
import seaborn as sns
import Swarm
import Environment
from SALib.sample import saltelli

sns.set_style('whitegrid')
iim = 5


def SimMidges(j, dps, eip):
    midgedeerratio = 100  # Midge/deer ratio

    deerpop = 100
    midgepop = deerpop * midgedeerratio

    midges = np.full(midgepop, False)
    midges[0:iim] = True  # Let some midges be infected with BTV

    deerinf = np.full(deerpop, False)  # Entire deer population is naive to BTV

    envir = Environment.Envir(length=1000)
    deer = Swarm.DeerSwarm(envir=envir, size=deerpop, infected=deerinf)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer, infected=midges, dps=dps, eip=eip)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    steps = 300 * 60  # Total number of steps for the simulation

    print("Moving swarm...")
    for i in range(steps):
        swrm.move(dt)

        if i % 300 == 0:
            print("Day", i // 300)

    print("Simulation finished")

    # print("Saving results...")
    # swrm.writetocsv(trial=j)
    # print("Results saved")

    return swrm.deerswarm.totalinfecteddeer[-1]


problem = {
    'num_vars': 2,
    'names': ['dps', 'eip'],
    'bounds': [
        [0.6, 0.9],
        [10, 20]
    ]
}

params = saltelli.sample(problem, 16)

for i in range(3):
    results = np.empty(shape=(params.shape[0], params.shape[1]+1))
    for j, X in enumerate(params):
        dps, eip = X
        # results[j] = SimMidges(i, dps, eip)
        results[j] = [dps, eip, SimMidges(i, dps, eip)]
        print(j, results[j])

    np.savetxt(fname='Results/IIM' + str(iim) + '/Trial' + str(i) + '.csv', X=results, delimiter=',', newline='\n')