import numpy as np
import seaborn as sns
import Swarm
import Environment
import threading
import time
import csv

sns.set_style('whitegrid')


def SimMidges(j, dps, eip):
    midgehostratio = 100  # Midge/host ratio

    hostpop = 100
    midgepop = hostpop * midgehostratio

    midges = np.full(midgepop, False)
    midges[0:5] = True  # Let some midges be infected with BTV

    hostinf = np.full(hostpop, False)  # Entire host population is naive to BTV

    envir = Environment.Envir(length=1000)
    host = Swarm.HostSwarm(envir=envir, size=hostpop, infected=hostinf)
    swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, hostswarm=host, infected=midges, dps=dps, eip=eip)
    dt = 60  # Step the simulation every 60 seconds (1 minute)
    days = 60  # Length in days of the simulation
    daylength = 300  # Number of steps in each day
    steps = daylength * days  # Total number of steps for the simulation

    print("Moving swarm...")
    for i in range(steps):
        swrm.move(dt)

        if i % 300 == 0:
            print("Day", i // 300)

    print("Simulation finished")

    return swrm.hostswarm.totalinfectedhost[-1]


params = []
for i in range(21):
    for j in range(21):
        params.append((0.6 + 0.015 * i, 10 + 0.5 * j))

print(params)


def CalculateHeatMap(i):
    for j in range(len(params)):
        with open('Results/HeatMap/Trial' + str(i) + '.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            dps, eip = params[j]
            print('Thread', i, 'Trial', j, 'in progress')

            writer.writerow([dps, eip, SimMidges(i, dps, eip)])


threadlist = []
for i in range(10):
    threadlist.append(threading.Thread(target=CalculateHeatMap, args=(i,)))

for t in threadlist:
    t.start()

for t in threadlist:
    t.join()
