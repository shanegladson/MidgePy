import numpy as np

""" This is the main swarm class, where all midges are simulated. This class holds all attributes of the midges and will
be responsible for moving the deer during its move function as well. Moving the time is done by calling the move() method
so be sure not to move the deer on their own! This feature will be added later.
"""


class MidgeSwarm:

    def __init__(self, envir, deerswarm, size=100, positions='random', infected='random'):

        self.step = 0  # Initialize the step counter
        self.size = size  # Define the population size of the swarm object
        self.pos_history = []  # Begin a list that tracks the history of midge positions
        self.activeflightvelocity = 0.5  # (m/s) Define the average active velocity of a midge per second
        self.roamflightvelocity = 0.13  # (m/s) Define the average roaming velocity of a midge per second
        self.detectiondistance = 300  # (m) Define the distance at which the midges can detect the deer
        self.bitethresholddistance = 3  # (m) Define the distance at which a midge must be in order to bite the deer
        self.iip = 21  # (days) Define the intrinsic incubation period (IIP)
        # self.fed = np.random.choice([True, False], size=self.size)  # Define an array for whether the midges have fed or not
        self.midgebitesperstep = []  # Keep track of the midge bites each time step
        self.totalinfectedmidges = []  # Keep track of the total number of infected midges
        self.btvincubating = np.full(self.size,
                                     False)  # Define the array that tracks whether BTV is incubating inside the midge
        self.envir = envir  # Attach the environment object to the swarm class
        self.deerswarm = deerswarm  # The midge swarm class will take the deer swarm to know the locations and attributes of each host
        self.daylength = 300  # The length in minutes of a single day (note it is not the entire day, only the length of each period simulated TODO: UPDATE DAY LENGTH
        self.timeoffeeding = np.random.randint(-self.daylength, 0,
                                               self.size)  # List to keep track of the time when each midge has fed
        self.randomvector = self.generate_random_vector()  # Array of random vector where the midges travel, updates every few steps

        # Create a random positions array for the midges if desired, otherwise it is defined
        if positions == 'random':
            self.positions = np.random.uniform(low=0.0, high=envir.length, size=(self.size, 2))
        else:
            self.positions = positions

        # Create a random array of which midges are infected if desired, otherwise it is defined
        if infected == 'random':
            self.infected = np.random.choice([True, False], self.size, p=[0.02, 0.98])
            # self.infected = np.full(self.size, False)
        else:
            self.infected = infected

    # The step function that calculates all movement (dt is given in seconds)
    def move(self, dt=1):

        # A new random vector is generated every 30 minutes for the midges to travel in
        if self.step % 30 == 0:
            self.randomvector = self.generate_random_vector()

        # Calculate which midges have fed lately by tracking when the last bloodmeal was
        self.fed = ~(np.abs(self.timeoffeeding - self.step) > self.daylength)

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

        # print(detectinghost.sum(), "midges detecting deer")

        # Calculate the new positions by using the flightvelocity variable, don't even ask me how it works bc idk
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
        self.feed(closestdeer)

        # Append the position history to pos_history
        self.pos_history.append(self.get_positions())

        # Increment the step counter
        self.step += 1

    # Returns the matrix of vectors from every midge to every deer (midges x deer x 2) size
    # TODO: COULD OPTIMIZE WITH NUMPY ARRAYS
    def calculate_target_matrix(self):
        matrix = []
        for m in self.get_positions():
            vectors = self.deerswarm.get_positions() - m
            matrix.append(vectors)
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

    # Random movement function that selects random direction uniformly and movement length on a normal curve,
    # sigma will be defined
    def generate_random_vector(self, sigma=1.0):

        # Calculate the angles from a uniform distribution, find the x- and y-components
        randomwalkangles = np.random.uniform(0, 2 * np.pi, self.size)
        xcomponent = np.cos(randomwalkangles)
        ycomponent = np.sin(randomwalkangles)
        randomwalkvector = np.vstack((xcomponent, ycomponent)).T

        return randomwalkvector

    def feed(self, closestdeer):

        # Find which midges are close enough to the host to bite them and they have not recently fed
        feedingmidges = (closestdeer < self.bitethresholddistance) & ~self.fed

        # The midges will become infected if they are feeding and the closest deer is infected, do the same for the deer
        self.infected = np.logical_or(self.infected, np.random.choice([True, False], self.size, p=[0.5, 0.5]) * (
                feedingmidges & self.deerswarm.infected[closestdeer]))

        # TODO: COULD PROBABLY MAKE THIS MORE EFFICIENT
        infectedprob = np.random.choice([True, False], self.size, p=(0.5, 0.5))
        # Track which deer are infected
        for i in range(self.size):
            # The deer can be infected with a 90% chance from a single bite from an infected midge
            self.deerswarm.infected[closestdeer[i]] = np.logical_or(
                (feedingmidges[i] & self.infected[i]) * infectedprob[i],
                self.deerswarm.infected[closestdeer[i]])

        # print(self.deerswarm.infected)
        # TODO: THE PROGRAM ONLY WORKS IF THIS LINE IS HERE BUT I DON'T KNOW WHY
        # self.fed = np.logical_or(self.fed, feedingmidges)
        self.fed[feedingmidges] = True

        # Update time of feeding to the current step for the midges which have just fed
        self.timeoffeeding[feedingmidges] = self.step

        # Append the midge bites for this time step and total infected midges
        self.midgebitesperstep.append(feedingmidges.sum())
        self.totalinfectedmidges.append(self.infected.sum()/self.size)
        self.deerswarm.totalinfecteddeer.append(self.deerswarm.infected.sum()/self.deerswarm.size)

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

    # Move function that is called by the MidgeSwarm class, DO NOT CALL!!!
    def move(self):
        print("Moving deer...")

    # TODO: Add deer movement functions

    # Returns the numpy array of positions
    def get_positions(self):
        return self.positions
