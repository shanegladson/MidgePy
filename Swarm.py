import numpy as np
import csv
from numba import jit, njit

""" This is the main swarm class, where all midges are simulated. This class holds all attributes of the midges and will
be responsible for moving the deer during its move function as well. Moving the time is done by calling the move() method
so be sure not to move the deer on their own! This feature will be added later.
"""


class MidgeSwarm:

    def __init__(self, envir, deerswarm, size=100, infected='random', dps=0.75, eip=21):

        self.step = 0  # Initialize the step counter
        self.size = size  # Define the population size of the swarm object
        self.pos_history = []  # Begin a list that tracks the history of midge positions
        self.activeflightvelocity = 0.50  # (m/s) Define the average active velocity of a midge per second
        self.roamflightvelocity = 0.13  # (m/s) Define the average roaming velocity of a midge per second
        self.detectiondistance = 300  # (m) Define the distance at which the midges can detect the deer
        self.bitethresholddistance = self.activeflightvelocity  # (m) Define the distance at which a midge must be in order to bite the deer
        self.eip = eip  # (days) Define the extrinsic incubation period (EIP)
        self.midgebitesperstep = []  # Keep track of the midge bites each time step
        self.totalinfectedmidges = []  # Keep track of the total number of infected midges
        self.infecteddeaths = []  # Keep track of the number of infected midges that die each step (only if midgedeath is True)
        self.uninfecteddeaths = []  # Keep track of the number of uninfected midges that die each step (only if midgedeath is True)
        self.btvincubating = np.full(self.size,
                                     False)  # Define the array that tracks whether BTV is incubating inside the midge
        self.incubationstarttime = np.full(self.size,
                                           0)  # Create an array that tracks when midges begin incubation for BTV
        self.envir = envir  # Attach the environment object to the swarm class
        self.deerswarm = deerswarm  # The midge swarm class will take the deer swarm to know the locations and attributes of each host
        self.daylength = 300  # The length in minutes of a single day (note it is not the entire day, only the length of each period simulated
        self.biterate = 2 * self.daylength  # This variable determines how often a midge is expected to bite a deer
        self.timeoffeeding = np.random.randint(-self.biterate, 0,
                                               self.size)  # List to keep track of the time when each midge has fed
        self.pVtoH = 0.90  # Probability of transmission of BTV from a vector to the host
        self.pHtoV = 0.14  # Probability of transmission of BTV from a host to the vector

        self.midgedeath = True  # Enable this if you would like to simulate midges dying and being replaced by new ones
        self.dps = dps  # Daily Probability of Survival. Only enable if self.midgedeath is true

        # Create a random positions array for the midges if desired, otherwise it is defined
        self.positions = np.random.uniform(low=0.0, high=envir.length, size=(self.size, 2))

        self.randomvector = generate_random_vector(self.envir.length, self.size, self.positions)  # Array of random vector where the midges travel, updates every few steps

        # Create a random array of which midges are infected if desired, otherwise it is defined
        if infected == 'random':
            self.infected = np.random.choice([True, False], self.size, p=[0.01, 0.99])
        else:
            self.infected = infected

    # The step function that calculates all movement (dt is given in seconds)
    def move(self, dt=1):

        self.totalinfectedmidges.append(self.infected.sum())

        # Update the infected midges to be those that have completed their EIP
        self.infected = np.logical_or(self.infected, np.logical_and(self.incubationstarttime != 0, np.abs(
            self.incubationstarttime - self.step) >= self.daylength * self.eip))

        self.deerswarm.infected = np.logical_or(self.deerswarm.infected, np.logical_and(
            self.deerswarm.incubationstarttime != 0,
            np.abs(self.deerswarm.incubationstarttime - self.step) >= self.daylength * self.deerswarm.incubationtime))

        # Move the deer once every day
        if self.step % self.daylength == 0:
            self.deerswarm.move()

            # Replace some midges once per day if self.midgedeath is enabled
            if self.midgedeath:
                survivingmidges = np.random.choice([True, False], self.size, p=[self.dps, 1 - self.dps])
                newpositions = np.random.uniform(low=0.0, high=self.envir.length, size=(self.size, 2))

                self.infecteddeaths.append(np.sum(~survivingmidges & self.infected))
                self.uninfecteddeaths.append(np.sum(~survivingmidges & ~self.infected))

                # Give the midges new positions and reset all other parameters
                self.btvincubating *= survivingmidges
                self.infected *= survivingmidges
                self.incubationstarttime *= survivingmidges
                survivingmidges = np.expand_dims(survivingmidges, 1)
                self.positions = self.positions * survivingmidges + newpositions * (~survivingmidges)

        # A new random vector is generated every 30 minutes for the midges to travel in
        if self.step % 30 == 0:
            self.randomvector = generate_random_vector(self.envir.length, self.size, self.positions)

        # Calculate which midges have fed lately by tracking when the last bloodmeal was
        self.fed = ~(np.abs(self.timeoffeeding - self.step) > self.biterate)

        # Find the matrix of vectors from each midge to each deer (self.size x self.deerswarm.size matrix)
        targetmatrix = self.calculate_target_matrix()

        # Calculate the matrix of distances from each midge to each deer and find the closest deer
        distancematrix = np.linalg.norm(targetmatrix, axis=2)
        closestdeer = np.argmin(distancematrix, axis=1)

        # Find the directions for each midge by finding vector from the closest deer
        midgedirections = np.empty(shape=(self.size, 2), dtype=float)
        for i in range(closestdeer.size):
            midgedirections[i] = targetmatrix[i, closestdeer[i]]

        # Calculate the distance that each midge must travel to reach the closest deer
        hostdistances = np.linalg.norm(midgedirections, axis=1)

        # The list of midges that are within the detection distance of their closest host and have not fed
        detectinghost = (hostdistances < self.detectiondistance) & ~self.fed

        # Calculate the new positions by using the flightvelocity variable
        self.positions = self.get_positions() + self.activeflightvelocity * dt * ((np.expand_dims(detectinghost, 1) *
                                                                                   np.divide(midgedirections,
                                                                                             np.expand_dims(
                                                                                                 hostdistances, 1),
                                                                                             out=np.zeros_like(
                                                                                                 midgedirections),
                                                                                             where=np.expand_dims(
                                                                                                 hostdistances,
                                                                                                 1) != 0)) + (
                                                                                          self.randomvector * ~np.expand_dims(
                                                                                      detectinghost, 1)))

        # Calculate which midges will feed and the results of their feeding
        self.feed(closestdeer, hostdistances, dt)

        # Append the position history to pos_history
        # self.pos_history.append(self.get_positions())

        # Increment the step counter
        self.step += 1

    # Returns the matrix of vectors from every midge to every deer (midges x deer x 2) size
    def calculate_target_matrix(self):
        matrix = np.empty(shape=(self.size, self.deerswarm.size, 2))
        pos = self.deerswarm.get_positions()

        # TODO: VERIFY THAT THIS WORKS
        for i in range(self.deerswarm.size):
            vectors = self.get_positions() - pos[i]
            matrix[:, i] = vectors

        return np.array(matrix)

    # Returns the numpy array of positions
    def get_positions(self):
        return self.positions

    # Returns the numpy array of infected midges
    def get_infected(self):
        return self.infected

    # Returns the DeerSwarm object
    def get_deerswarm(self):
        return self.deerswarm

    # Returns the full position history of the midges
    def get_full_pos_history(self):
        return [*self.pos_history, self.get_positions()]

    def feed(self, closestdeer, hostdistances, dt):

        # Find which midges are close enough to the host to bite them and they have not recently fed
        feedingmidges = determinefeedingmidges(self.size, hostdistances, closestdeer, self.bitethresholddistance, dt, self.fed)

        # The midges will begin BTV incubation if they are feeding and the closest deer is infected, do the same for the deer
        self.newincubation = np.random.choice([True, False], self.size, p=[self.pHtoV, 1 - self.pHtoV]) * (
                feedingmidges & self.deerswarm.infected[closestdeer])

        self.incubationstarttime[self.newincubation] = self.step

        # Create the probability of infection array that determines which deer will become infected if bitten during this timestep
        infectedprob = np.random.choice([True, False], self.size, p=(self.pVtoH, 1 - self.pVtoH))
        # Track which deer become inoculated (if they are bitten, midge is infected, probability is favorable, and have not already been inoculated)
        # for i in range(self.size):
        #     if feedingmidges[i] & self.infected[i] & infectedprob[i] & (self.deerswarm.incubationstarttime[closestdeer[i]] == 0):
        #         # The deer can be infected with probability infectedprob from a single bite from an infected midge
        #         self.deerswarm.incubationstarttime[closestdeer[i]] = self.step

        self.deerswarm.incubationstarttime = determineincubation(self.step, self.size, feedingmidges, self.infected, infectedprob, self.deerswarm.incubationstarttime, closestdeer)

                # Update time of feeding to the current step for the midges which have just fed
        self.timeoffeeding[feedingmidges] = self.step

        # Append the midge bites for this time step and total infected midges
        self.midgebitesperstep.append(feedingmidges.sum())
        self.deerswarm.totalinfecteddeer.append(self.deerswarm.infected.sum())

    def writetocsv(self, trial, fname='Results/midgesim'):

        fname = fname + 'IIM' + str(int(self.totalinfectedmidges[0])) + 'Trial' + str(trial) + '.csv'

        with open(fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ['Step', 'Day', 'Infected Midges', 'Infected Midges %', 'Infected Deer', 'Infected Deer %', 'Midge Bites',
                 'VF', 'VR', 'DD', 'EIP', 'PVTH', 'PHTV', 'DPS', 'PD', 'BR',
                 'MDR', 'IIM', 'Infected Deaths', 'Uninfected Deaths'])
            for i in range(self.step):
                if self.midgedeath:
                    writer.writerow([i, i/float(self.daylength), self.totalinfectedmidges[i], self.totalinfectedmidges[i] / self.size * 100,
                                     self.deerswarm.totalinfecteddeer[i],
                                     self.deerswarm.totalinfecteddeer[i] / self.deerswarm.size * 100,
                                     self.midgebitesperstep[i], self.activeflightvelocity,
                                     self.roamflightvelocity, self.detectiondistance, self.eip, self.pVtoH,
                                     self.pHtoV, self.dps, self.deerswarm.size, self.biterate,
                                     self.size // self.deerswarm.size,
                                     self.totalinfectedmidges[0], self.infecteddeaths[i // self.daylength],
                                     self.uninfecteddeaths[i // self.daylength]])

                else:
                    writer.writerow([i, i/float(self.daylength), self.totalinfectedmidges[i], self.totalinfectedmidges[i] / self.size * 100,
                                     self.deerswarm.totalinfecteddeer[i],
                                     self.deerswarm.totalinfecteddeer[i] / self.deerswarm.size * 100,
                                     self.midgebitesperstep[i], self.activeflightvelocity, self.roamflightvelocity,
                                     self.detectiondistance, self.eip, self.pVtoH,
                                     self.pHtoV, self.dps, self.deerswarm.size, self.biterate,
                                     self.size // self.deerswarm.size,
                                     self.totalinfectedmidges[0]])

        # with open('positions.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     header = ['Step', 'Midge 0 X', 'Midge 0 Y']
        #     writer.writerow(header)
        #     for i in range(self.step):
        #         writer.writerow([i, self.pos_history[i][0][0], self.pos_history[i][0][1]])
        return


class DeerSwarm:

    def __init__(self, envir, size=50, positions='random', infected='random', steplength=1.0):

        # Define the population size of the swarm object
        self.size = size

        # Define the average step length of a deer per time step
        self.avgsteplength = steplength

        # Keep track of the total number of infected deer
        self.totalinfecteddeer = []

        # Attach the environment object to the swarm class
        self.envir = envir

        self.incubationtime = 2

        self.incubationstarttime = np.full(self.size,
                                           0)  # Create an array that tracks when midges begin incubation for BTV

        # Create a random positions array for the deer if desired, otherwise it is defined
        if positions == 'random':
            self.positions = np.random.uniform(low=0.0, high=envir.length, size=(self.size, 2))
        else:
            self.positions = positions

        # Create a random array of which deer begin infected if desired, otherwise it is defined
        if infected == 'random':
            self.infected = np.full(self.size, False)
            # self.infected = np.random.choice([True, False], self.size, p=[0.2, 0.8])
        else:
            self.infected = infected

    # Move function that is called by the MidgeSwarm class, generates a new set of points for the deer (random)
    def move(self):
        self.positions = np.random.uniform(low=0.0, high=self.envir.length, size=(self.size, 2))

    # Returns the numpy array of positions
    def get_positions(self):
        return self.positions

# Random movement function
def generate_random_vector(length, size, positions):
    # Creates a vector from the midge to a random position within the domain, then the midge will follow that vector
    newvectors = np.random.uniform(low=0.0, high=length, size=(size, 2)) - positions
    newvectors /= np.expand_dims(np.linalg.norm(newvectors, axis=1), axis=1)

    return newvectors

@jit
def determinefeedingmidges(length, hostdistances, closestdeer, bitethresholddistance, dt, fed):
    feedingmidges = np.full(length, False)
    for i in range(length):
        feedingmidges[i] = (hostdistances[closestdeer[i]] < bitethresholddistance * dt) & ~fed[i]

    return feedingmidges

@jit
def determineincubation(step, length, feedingmidges, infected, infectedprob, deerswarmincubationstarttime, closestdeer):
    for i in range(length):
        if feedingmidges[i] & infected[i] & infectedprob[i] & (deerswarmincubationstarttime[closestdeer[i]] == 0):
            # The deer can be infected with probability infectedprob from a single bite from an infected midge
            deerswarmincubationstarttime[closestdeer[i]] = step

    return deerswarmincubationstarttime