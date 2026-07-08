import numpy as np

class SingleAgent:
    def __init__(self, K, policy="UCB", param=4.0):
        self.K = K
        self.policy = policy
        self.param = param
        self.pulls = np.zeros(K)
        self.rewards = np.zeros(K)
        self.t = 0

    def choose(self):
        self.t += 1
        unplayed = np.where(self.pulls == 0)[0]
        if len(unplayed) > 0:
            return unplayed[0]
            
        if self.policy == "UCB":
            indices = (self.rewards / self.pulls) + np.sqrt((self.param * np.log(self.t)) / self.pulls)
            return np.argmax(indices)
        elif self.policy == "Epsilon-Greedy":
            eps = min(1.0, self.param / self.t)
            if np.random.rand() < eps:
                return np.random.randint(self.K) 
            else:
                return np.argmax(self.rewards / self.pulls) 
        elif self.policy == "Thompson-Sampling":
            samples = np.random.beta(self.rewards + 1, (self.pulls - self.rewards) + 1)
            return np.argmax(samples)

    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward

class MaliciousAgent:
    def __init__(self, K):
        self.K = K
    def get_recommendation_to_send(self):
        return np.random.randint(self.K)

class GossipingAgent:
    def __init__(self, K, sticky_set, beta, policy="UCB", param=4.0):
        self.K = K
        self.sticky_set = list(sticky_set)
        self.beta = beta
        self.policy = policy
        self.param = param
        self.pulls = np.zeros(K)
        self.rewards = np.zeros(K)
        self.pulls_in_phase = np.zeros(K)
        self.t = 0
        self.j = 1
        self.A_j = int(np.ceil(self.j ** self.beta))
        
        available = list(set(range(K)) - set(self.sticky_set))
        np.random.shuffle(available)
        self.U = available[0]
        self.L = available[1]
        self.active_set = self.sticky_set + [self.U, self.L]
        # OPTIMASI: SIMPAN SEBAGAI NUMPY ARRAY PERMANEN
        self.active_arms = np.array(self.active_set) 

    def choose(self):
        self.t += 1
        p = self.pulls[self.active_arms]
        r = self.rewards[self.active_arms]
        
        unplayed_idx = np.where(p == 0)[0]
        if len(unplayed_idx) > 0:
            return self.active_arms[unplayed_idx[0]]
            
        if self.policy == "UCB":
            indices = (r / p) + np.sqrt((self.param * np.log(self.t)) / p)
            return self.active_arms[np.argmax(indices)]
        elif self.policy == "Epsilon-Greedy":
            eps = min(1.0, self.param / self.t)
            if np.random.rand() < eps:
                return self.active_arms[np.random.randint(len(self.active_arms))]
            else:
                return self.active_arms[np.argmax(r / p)]
        elif self.policy == "Thompson-Sampling":
            samples = np.random.beta(r + 1, (p - r) + 1)
            return self.active_arms[np.argmax(samples)]

    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.pulls_in_phase[arm] += 1

    def check_end_phase(self):
        return self.t == self.A_j

    def get_recommendation_to_send(self):
        p_phase = self.pulls_in_phase[self.active_arms]
        return self.active_arms[np.argmax(p_phase)]

    def end_phase(self, received_arm):
        if received_arm not in self.active_set:
            non_sticky = np.array([self.U, self.L])
            non_sticky_pulls = self.pulls_in_phase[non_sticky]
            self.U = non_sticky[np.argmax(non_sticky_pulls)]
            self.L = received_arm
            self.active_set = self.sticky_set + [self.U, self.L]
            # UPDATE NUMPY ARRAY HANYA SAAT FASE BERAKHIR
            self.active_arms = np.array(self.active_set) 
            
        self.pulls_in_phase.fill(0)
        self.j += 1
        self.A_j = int(np.ceil(self.j ** self.beta))