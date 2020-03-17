import numpy as np
import matplotlib.pyplot as plt
from Non_Stationary.Non_Stationary_Environment import Non_Stationary_Environment
from Stationary.TS_Learner import *
from Non_Stationary.SWTS_Learner import SWTS_Learner

n_arms = 4
# For every phase (every row) the value of the arms is expressed (the p parameter of the Bernoulli)
p = np.array([[0.15, 0.1, 0.2, 0.35],
              [0.35, 0.21, 0.2, 0.35],
              [0.5, 0.1, 0.1, 0.15],
              [0.8, 0.21, 0.1, 0.15]])

T = 400  # time horizon
n_experiments = 1000
ts_rewards_per_experiment = []  # needed to collect the rewards for each experiment
swts_rewards_per_experiment = []    # needed to collect the rewards for each experiment
window_size = int(np.sqrt(T))

for e in range(n_experiments):
    ts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
    ts_learner = TS_Learner(n_arms)

    swts_env = Non_Stationary_Environment(n_arms, probabilities=p, horizon=T)
    swts_learner = SWTS_Learner(n_arms, window_size)

    # Note that we need two different environments since each environment has its own variables
    # This implies that everything is equal for the two environments except for np.random.binomial(1, p) and np.random.beta()
    # np.random.binomial(1, p) can be either 0 or 1, and it is random.
    # However, considering the experiments are many, the average result will be p, for both.

    for t in range(T):
        # Pull an arm according to the beta distribution
        pulled_arm = ts_learner.pull_arm()
        # Observe its reward
        reward = ts_env.round(pulled_arm)
        # Update the observations (that is save the just observed reward) and update the parameters of the beta distribution
        ts_learner.update(pulled_arm, reward)

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

    # For each experiment save the collected rewards for each t
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    swts_rewards_per_experiment.append(swts_learner.collected_rewards)

ts_instantaneous_regret = np.zeros(T)
swts_instantaneous_regret = np.zeros(T)
n_phases = len(p)
phases_len = int(T / n_phases)  # all phases are assumed to be of the same length

# Compute the optimum arm for each phase (getting the max for each row -> axis=1)
opt_per_phase = p.max(axis=1)
optimum_per_round = np.zeros(T)

# Compute the mean of the rewards for every t over all the experiments
avg_regrets_ts = np.mean(ts_rewards_per_experiment, axis=0)
avg_regrets_swts = np.mean(swts_rewards_per_experiment, axis=0)
for i in range(n_phases):
    # For each phase calculate the optimal arm per round (for every t of the phase) setting it equal to the optimal for the phase
    optimum_per_round[i * phases_len:(i + 1) * phases_len] = opt_per_phase[i]
    # For each phase calculate the instantaneous regret for TS and SWTS
    # instantaneous_regret_t = opt_t - avg_regret_t
    ts_instantaneous_regret[i * phases_len:(i + 1) * phases_len] = opt_per_phase[i] - avg_regrets_ts[i * phases_len:(i + 1) * phases_len]
    swts_instantaneous_regret[i * phases_len:(i + 1) * phases_len] = opt_per_phase[i] - avg_regrets_swts[i * phases_len:(i + 1) * phases_len]

plt.figure(0)
plt.ylabel("Reward")
plt.xlabel("t")
plt.plot(avg_regrets_ts, 'r')
plt.plot(avg_regrets_swts, 'b')
plt.plot(optimum_per_round, '--k')
plt.legend(["TS", "SW-TS", "Optimum"])
plt.show()

plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(ts_instantaneous_regret, axis=0), 'r')
plt.plot(np.cumsum(swts_instantaneous_regret, axis=0), 'b')
plt.legend(["TS", "SW-TS"])
plt.show()
