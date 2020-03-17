import matplotlib.pyplot as plt
from Stationary.Environment import *
from Stationary.TS_Learner import *
from Stationary.Greedy_Learner import *

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

# Regret = T*opt - sum_t(rewards_t)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")


# Calculate the instantaneous regret for each t for each experiment
# Note: regret_t = optimal_arm_value - pulled_arm_value
# Note: a positive regret is "bad", a negative regret is "good"
regrets = opt - ts_rewards_per_experiment

# Calculate the average regret for each iteration t
# Note that we are conducting n_experiments so we need the average over all the experiments for each iteration t
# Note that axis=0 means that you are averaging over each iteration t (over the column)
avg_regrets = np.mean(regrets, axis=0)

# Avg_Regret = T*opt - sum_t(value_t)
# Note that we have already calculated the average regret (opt-value)
# So we only need to cumulatively sum the array containing the avg_regret for each itearation t
# np.cumsum(array) returns an array with item at position i equal to the sum of the items at previous positions (pos i included)
avg_regret = np.cumsum(avg_regrets)
# Plot
plt.plot(avg_regret, 'r')

# The same is done for Greedy_Learner
plt.plot(np.cumsum(np.mean(opt-gr_rewards_per_experiment, axis=0)), 'g')

plt.legend(["TS", "Greedy"])
plt.show()




























