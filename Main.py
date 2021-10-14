import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Swarm
import Environment

sns.set_style('whitegrid')

midgedeerratio = 20  # Midge/deer ratio

deerpop = 100
midgepop = deerpop * midgedeerratio

midges = np.full(midgepop, False)
midges[0:5] = True  # Let some midges be infected with BTV

deerinf = np.full(deerpop, False)
deerinf[0:70] = False  # Let some midges be infected with BTV

envir = Environment.Envir(length=1000)
deer = Swarm.DeerSwarm(envir=envir, size=deerpop, infected=deerinf)
swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer, infected=midges)
dt = 60  # Step the simulation every 60 seconds (1 minute)
steps = 300 * 5  # Total number of steps for the simulation


print("Moving swarm...")
for i in range(steps):
    swrm.move(dt)

    if i % 300 == 0:
        print("Step", i)

swrm.writetocsv()
