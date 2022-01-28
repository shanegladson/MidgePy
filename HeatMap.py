import numpy as np
import seaborn as sns
import Swarm
import Environment
from SALib.sample import saltelli

sns.set_style('whitegrid')


def SimMidges(j, dps, eip):
    midgedeerratio = 100  # Midge/deer ratio

    deerpop = 100
    midgepop = deerpop * midgedeerratio

    midges = np.full(midgepop, False)
    midges[0:5] = True  # Let some midges be infected with BTV

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


params = []
for i in range(20):
    for j in range(20):
        params.append(((0.6+0.015*i), 10+0.5*j))

print(params)

for i in range(7, 10):
    results = np.empty(shape=(np.array(params).shape[0], np.array(params).shape[1]+1))
    for j in range(len(params)):
        dps, eip = params[j]
        # results[j] = SimMidges(i, dps, eip)
        results[j] = [dps, eip, SimMidges(i, dps, eip)]
        print(j, results[j])

    np.savetxt(fname='Results/HeatMap/Trial' + str(i) + '.csv', X=results, delimiter=',', newline='\n')


