from bandit_environments import *
from bandit_agents import *
from bandit_policies import *
from bandit_testrunner import *
"""
We set up the bandit enviorment as k = 10 ,u = 10 sigma =1.the enviorment  is stationary,
This time we compare two differnt agent, one is greedy eplison 1% agent another is UCB agent.
Then we have 1000 timesteps,run for 2000 times and eventually get two plot.

"""

bandit = BanditEnvironment(k = 10, mu = 0, sigma = 1)
egreedy_1perc = BanditEpsilonGreedyPolicy(epsilon = 0.01)
ucb_c2 =  BanditUCBPolicy(C = 2)

agent1 = SampleAverageBanditAgent(policy = egreedy_1perc , bandit_env = bandit)
agent2 = SampleAverageBanditAgent(policy = ucb_c2, bandit_env = bandit)

agents = [agent1, agent2]

runner = BanditTestRunner(agents = agents, bandit_env = bandit)

r, o = runner.perform_runs(timesteps = 1000, runs = 2000)

runner.visualize_results(
    save_filename = "question1",
        title = "Epsilon-Greedy 10% vs. Gradient Bandit",
        rewards_histories = r,
        optimal_action_histories = o
        )