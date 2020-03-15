import numpy as np
import matplotlib.pyplot as plt
from Pricing.Environment import *
from Pricing.TS_Learner import *
from Pricing.Greedy_Learner import *

n_arms = 4
p = np.array([0.15, 0.1, 0.1,
              0.35])  # probabilities for each arm (probability of obtaining 1 as a sample (Bernoulli lies in {0,1})
opt = p[3]  # This is the optimal arm (0.35 is the greatest) --> My guess

T = 300  # Time Horizon

n_experiments = 1000    # number of experiments

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range(n_experiments):
    print("Experiment ", e)
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms)
    gr_learner = Greedy_Learner(n_arms)
    for t in range(T):
        # TS Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
# Regret = T*opt - sum rewards
plt.plot(np.cumsum(np.mean(opt-ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt-gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()




























