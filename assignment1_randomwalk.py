from bandit_environments import *
from bandit_agents import *
from bandit_policies import *
from bandit_testrunner import *

"""
We set up the bandit enviorment as k = 10 ,u = 10 sigma =1.it is non stationary,
because each time the agent take a action against the enviorment, we add a incremental term N(0,0.01)in q*
to make sure that the q* is not stationary.And then we try to construct two agent they both use the greedy e
plison 1% but one of them use sample aveage but another take fixed learning rate. we set the timestep =10000
and runs = 500 and compare those two agents behave. and then we try to more disturbation N(0,0.1),others conditions
keep the same, and we compare those two agents hebave in more non stationary situation.


"""


nonstabandit = BanditEnvironment(k = 10, mu = 0, sigma = 1)
egreedy_1perc = BanditEpsilonGreedyPolicy(epsilon = 0.01)

agent1 = SampleAverageBanditAgent(policy = egreedy_1perc , bandit_env = nonstabandit)
agent2 = ConstantstepAgent(policy = egreedy_1perc, bandit_env = nonstabandit,stepsize =0.1)

agents = [agent1, agent2]

runner = BanditTestRunner(agents = agents, bandit_env = nonstabandit)

r, o = runner.perform_runs(timesteps = 10000, runs = 500)

runner.visualize_results(
    save_filename = "quesiton2_large",
        title = "non stationary(large) Sample average(greedy 1%) vs. non stationary(large) constant step (greedy 1%) ",
        rewards_histories = r,
        optimal_action_histories = o
        )