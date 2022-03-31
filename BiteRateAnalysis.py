import numpy as np
import Swarm
import Environment
import threading


# THE PURPOSE OF THIS ANALYSIS IS TO UNDERSTAND HOW DPS AFFECTS BITING RATE

def SimMidges(j, dps):
    midgedeerratio = 100  # Midge/deer ratio

    deerpop = 100
    midgepop = deerpop * midgedeerratio

    midges = np.full(midgepop, True)  # ALL FIRST GENERATION MIDGES WILL BE INFECTED, NO OTHERS

    deerinf = np.full(deerpop, False)  # Entire deer population is naive to BTV

    envir = Environment.Envir(length=1000)
    deer = Swarm.DeerSwarm(envir=envir, size=deerpop, infected=deerinf)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer, infected=midges, dps=dps)
    swrm.pVtoH = 0  # Don't want to consider transmission to deer
    swrm.eip = 100  # Again just to be sure
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    steps = 0  # Track the total number of steps for the simulation

    print("Moving swarm...")
    # RUN UNTIL ALL INFECTED MIDGES (FIRST GEN) HAVE DIED
    while np.sum(swrm.infected) > 0 and steps <= (10 * 300):
        swrm.move(dt)

        steps += 1

    print("Simulation finished")

    print("Saving results...")
    swrm.writetocsv(trial=j, fname='Results/BiteRateAnalysis/AllInfected')
    print("Results saved")


threadlist = []
for i in np.arange(0, 1, 0.05):
    threadlist.append(threading.Thread(target=SimMidges, args=(1, i)))
    # SimMidges(1, dps=i)
    # print('DPS ' + str(i) + ' simulation completed')
for t in threadlist:
    t.start()

for t in threadlist:
    t.join()
