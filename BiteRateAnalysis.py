import numpy as np
import Swarm
import Environment


def SimMidges(j, dps):
    midgedeerratio = 100  # Midge/deer ratio

    deerpop = 100
    midgepop = deerpop * midgedeerratio

    midges = np.full(midgepop, False)  # NOT TRACKING INFECTED MIDGES FOR THIS ANALYSIS

    deerinf = np.full(deerpop, False)  # Entire deer population is naive to BTV

    envir = Environment.Envir(length=1000)
    deer = Swarm.DeerSwarm(envir=envir, size=deerpop, infected=deerinf)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer, infected=midges, dps=dps)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    steps = 300 * 10  # Total number of steps for the simulation

    print("Moving swarm...")
    for i in range(steps):
        swrm.move(dt)

        if i % 300 == 0:
            print("Day", i // 300)

    print("Simulation finished")

    print("Saving results...")
    swrm.writetocsv(trial=j, fname='Results/BiteRateAnalysis/')
    print("Results saved")


for i in np.arange(0, 1, 0.05):
    SimMidges(1, dps=i)
