import numpy as np
from tqdm import tqdm
from src.agents import SingleAgent, GossipingAgent, MaliciousAgent

def run_single_agent_baseline(means, stds, T, policy="UCB", explore_param=4.0, pos=0):
    K = len(means)
    best_mean = np.max(means)
    agent = SingleAgent(K, policy, explore_param)
    
    cumulative_regret = 0
    regret_history = []
    reward_history = []
    
    # mininterval=2.0 mencegah Terminal Windows menjadi macet (lag)
    for t in tqdm(range(T), desc=f"Single Agent ({policy})", position=pos, leave=True, mininterval=2.0):
        arm = agent.choose()
        
        # Fast Scalar Random Normal
        reward = np.random.normal(means[arm], stds[arm])
        reward = max(0.0, min(1.0, reward)) # Pengganti np.clip yang jauh lebih ringan
        
        agent.update(arm, reward)
        cumulative_regret += (best_mean - means[arm])
        regret_history.append(cumulative_regret)
        reward_history.append(reward)
        
    return np.array(regret_history), np.array(reward_history)


def run_multi_agent_system(means, stds, T, n_honest, m_malicious, beta, policy="UCB", explore_param=4.0, pos=0):
    K = len(means)
    best_mean = np.max(means)
    best_arm_idx = np.argmax(means)
    
    S_size = max(2, int((K / n_honest) * np.log(1/0.1)))
    all_arms = list(range(K))
    np.random.shuffle(all_arms)
    
    if best_arm_idx in all_arms: all_arms.remove(best_arm_idx)
    
    honest_agents = []
    for i in range(n_honest):
        if i == 0:
            sticky = [best_arm_idx] + all_arms[:S_size-1]
            all_arms = all_arms[S_size-1:]
        else:
            sticky = all_arms[:S_size]
            all_arms = all_arms[S_size:]
        honest_agents.append(GossipingAgent(K=K, sticky_set=sticky, beta=beta, policy=policy, param=explore_param))
        
    malicious_agents = [MaliciousAgent(K) for _ in range(m_malicious)]
    all_agents = honest_agents + malicious_agents
    
    cumulative_regret = 0
    regret_history = []
    reward_history = []
    
    desc_str = f"Multi-Agent (n={n_honest}, m={m_malicious})"
    # mininterval=2.0 mencegah Terminal Windows menjadi macet (lag)
    for t in tqdm(range(1, T + 1), desc=desc_str, position=pos, leave=True, mininterval=2.0):
        step_regret = 0
        step_reward = 0
        
        for agent in honest_agents:
            arm = agent.choose()
            
            # Fast Scalar Random Normal
            reward = np.random.normal(means[arm], stds[arm])
            reward = max(0.0, min(1.0, reward))
            
            agent.update(arm, reward)
            step_regret += (best_mean - means[arm])
            step_reward += reward
            
        cumulative_regret += (step_regret / n_honest)
        regret_history.append(cumulative_regret)
        reward_history.append(step_reward / n_honest)
        
        # Communication Phase
        for agent in honest_agents:
            if agent.check_end_phase():
                valid_targets = [a for a in all_agents if a != agent]
                target = np.random.choice(valid_targets)
                rec_arm = target.get_recommendation_to_send()
                agent.end_phase(rec_arm)
                
    return np.array(regret_history), np.array(reward_history)