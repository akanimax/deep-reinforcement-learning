""" Script for running the experiment with different number of
    bandits and different values of their rewards
"""

import EpsilonGreedy as eg
import matplotlib.pyplot as plt

# Define the parameters for tweaking
# ====================================================================
bandit_means = [37, 29, 54, 83, 27, 62]
epsilon = 0.1
n = 10000  # number of iterations
# Note that n should be an integer
# ====================================================================

# run the simulation
result = eg.simulate(*bandit_means, epsilon=epsilon, n=n, feed_back=True, feed_back_factor=1000)

# plot the result using matplotlib
plt.figure().suptitle("Simulation result")
plt.plot(result)
# plot the means for these values as a comparison
for bandit_mean in bandit_means:
    plt.plot([bandit_mean] * n)

# plot this on logscale
plt.xscale('log')

plt.show()

# done ...
