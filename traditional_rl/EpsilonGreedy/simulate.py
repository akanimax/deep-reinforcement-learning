""" Run the simulation of the multi-armed bandits
"""

import numpy as np

from EpsilonGreedy.bandit import Bandit


def simulate(*means, epsilon=0.1, n=10000, feed_back=True, feed_back_factor=100):
    """
    Run the simulation of the multi-armed bandit problem using the Epsilon_Greedy strategy

    args:
        *means: means for the bandits in the experimentation
            the number of mean arguments passed will determine
            the number of bandits

        epsilon: The epsilon value for the strategy

        n: number of iterations to simulate

        feed_back: printing feedback required or not

        feed_back_factor: number of iterations after which feedback message is displayed

    return:
        Cumulative_Average: List of average rewards received during the simulation
    """
    if epsilon > 1 or epsilon <= 0:
        raise ValueError("Value of epsilon must be in range (0, 1.0)")

    if n < 0:
        raise ValueError("Value of 'n' cannot be negative")

    data = []  # initialize to empty array
    bandits = [Bandit(m) for m in means]
    print("Running the experiment ... ")
    for cnt in range(n):
        sample = np.random.random_sample()
        if sample < epsilon:
            # explore (case)
            # select a random bandit
            bandit = np.random.choice(bandits)
        else:
            # exploit
            current_best = np.argmax([b.mean for b in bandits])
            bandit = bandits[current_best]
        # pull the bandit's arm and update the mean estimate
        reward = bandit.pull()
        bandit.update(reward)
        data.append(reward)

        if feed_back and (cnt+1) % feed_back_factor == 0:
            print("Ran %d iterations ..." % (cnt+1))

    # obtain the cumulative averages from the rewards
    cumulative_averages = np.cumsum(data) / np.arange(1, n+1)

    print("Experiment complete ...")

    # return the cumulative averages
    return cumulative_averages
