import numpy as np
import seaborn as sns
import Swarm
import Environment
import csv

sns.set_style('whitegrid')


def Outbreak(dps, eip, iim):
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

    for l in range(steps):
        swrm.move(dt)

        if np.sum(swrm.infected) == 0 and np.sum(swrm.deerswarm.incubationstarttime) == 0:
            return False
        elif np.sum(swrm.deerswarm.incubationstarttime) != 0:
            return True

    return False


numsims = 500
iim = 6
dps = np.linspace(0.0, 1.0, num=51, endpoint=True)
print(dps)
eip = 14
numoutbreaks = np.empty(shape=(dps.shape[0], 2), dtype=float)

for k in range(iim):
    for i in range(dps.shape[0]):
        success = np.empty(shape=(numsims), dtype=bool)
        for j in range(numsims):
            success[j] = Outbreak(dps=dps[i], eip=eip, iim=k+1)
            print('DPS:', dps[i], ' Simulation:', j, ' Outbreak:', success[j])

        numoutbreaks[i, 0] = dps[i]
        numoutbreaks[i, 1] = np.sum(success)

    print(numoutbreaks)
    with open('Results/IIM' + str(k+1) + '/OutbreakProbabilityLonger.csv', 'a') as file:
        writer = csv.writer(file)
        for row in numoutbreaks:
            writer.writerow(row)
