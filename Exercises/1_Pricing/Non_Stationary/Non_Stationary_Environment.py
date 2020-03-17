from Stationary.Environment import *


class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super(Non_Stationary_Environment, self).__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon

    def round(self, pulled_arm):
        n_phases = len(self.probabilities)
        phase_size = self.horizon / n_phases  # assume the phases are of the same duration
        current_phase = int(self.t / phase_size)  # 0 is the first phase

        p = self.probabilities[current_phase][pulled_arm]  # parameter of Bernoulli distribution
        self.t += 1
        return np.random.binomial(1, p)
