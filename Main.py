import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Swarm
import Environment

sns.set_style('whitegrid')

midgedeerratio = 30  # Midge/deer ratio

deerpop = 100
midgepop = deerpop * midgedeerratio

midges = np.full(midgepop, False)
midges[0:5] = True  # Let some midges be infected with BTV

envir = Environment.Envir(length=2500)
deer = Swarm.DeerSwarm(envir=envir, size=deerpop)
swrm = Swarm.MidgeSwarm(envir=envir, size=midgepop, deerswarm=deer, infected=midges)
dt = 60  # Step the simulation every 60 seconds (1 minute)
steps = 300 * 3  # Total number of steps for the simulation


print("Moving swarm...")
for i in range(steps):
    swrm.move(dt)

    if i % 300 == 0:
        print("Step", i)

swrm.writetocsv()

# sns.lineplot(x=range(steps), y=swrm.totalinfectedmidges, label='% Midge Infections')
# sns.lineplot(x=range(steps), y=swrm.deerswarm.totalinfecteddeer, label='% Deer Infections')
# plt.legend()
# plt.title("Midge and Deer Infections Over Time")
# plt.xlabel("Time (in minutes)")
# plt.ylabel("% Infected")
# plt.savefig('MidgeDeerInfections.svg', dpi=800)
# plt.show()
