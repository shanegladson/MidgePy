import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Swarm
import Environment

sns.set_style('whitegrid')

midgedeerratio = 500  # Midge/deer ratio

deerpop = 100
midgepop = midgedeerratio * deerpop

envir = Environment.Envir(length=800)
deer = Swarm.DeerSwarm(envir=envir, size=deerpop)
swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer)
dt = 60  # Step the simulation every 60 seconds (1 minute)
steps = 400  # Total number of steps for the simulation

print("Moving swarm...")
for i in range(steps):
    swrm.move(dt)

    if i % 10 == 0:
        print("Step", i)

sns.lineplot(x=range(steps), y=swrm.totalinfectedmidges, label='% Midge Infections')
sns.lineplot(x=range(steps), y=swrm.deerswarm.totalinfecteddeer, label='% Deer Infections')
plt.legend()
plt.title("Midge and Deer Infections Over Time")
plt.xlabel("Time (in minutes)")
plt.ylabel("% Infected")
plt.savefig('MidgeDeerInfections.svg', dpi=800)
plt.show()
