import numpy as np
import random
import sys
"""
Recall that policies are a class of things that choose actions. They represent
the computation that the agent carries out that outputs what to do.

As a result, policies are a great candidate for the Strategy design pattern.
Every policy follows the same general structure, and so we have them inherit
their structure from a single, higher-level Policy object.

Each new class is a variant on the Policy base framework. Just like how each new
iPhone is kind of different... but not really lol

Amazing for faster experimentation. Off we go!
"""

class BanditPolicy():
    """
    This is the base/parent policy class, not meant to be used directly.

    We place this at the top of the file so that anyone reading this code in 
    future - including future you - will know that to use any policy class, they
    mainly just need to read this one. Read one thing, and learn most of how
    all the others work. So convenient!!
    """

    # We don't need an init, since this is about the shared methods of choice.

    def argmax_with_random_tiebreaker(self, action_value_estimates):
        """
        Chooses the maximum of the provided action-value estimates,
        with ties broken randomly.

        Args:
            action_value_estimates: A numpy array containing action-value
            estimates.
        Returns:
            The index of the max element.
        """
        return np.random.choice(
            np.where( action_value_estimates == action_value_estimates.max())[0]
        )
    
    def choose_action(self):
        """
        This method is just here to be overridden, so it is 'empty'.
        """
        pass

class BanditEpsilonGreedyPolicy(BanditPolicy):
    """
    The epsilon-greedy action selection policy. An agent following the 
    epsilon-greedy policy will choose a random action with probability epsilon,
    and will greedily choose the best action (argmax) with probability 
    1 - epsilon. If multiple actions are tied for the best choice, ties are
    broken randomly.
    
    Attributes:
        epsilon: A value [0,1] that determines the probability that an agent will
        randomly choose an action at each timestep.
    """
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon
    
    def __str__(self):
        return "Epsilon-Greedy Policy: {e}".format(e = self.epsilon)
    
    def choose_action(self, agent_data):
        """
        This is where we override the method that's in the base class.

        Args:
            action_value_estimates: A numpy array containing the action-value
            estimates for a given bandit problem environment.
        Returns:
            action: The index of the chosen action.
        """
        action_value_estimates = agent_data["action_value_estimates"]
        roll = random.uniform(0,1)
        if roll <= self.epsilon:
            action = random.choice( list( range(0,len(action_value_estimates))))
        else:
            action = self.argmax_with_random_tiebreaker(action_value_estimates)
        return action
    

class BanditUCBPolicy(BanditPolicy):
    """
    Upper-Confidence-Bound Action Selection Policy
    The idea of this upper confidence bound (UCB) action selection is that the square-root
t   erm is a measure of the uncertainty or variance in the estimate of a’s value. The quantity
b   eing max’ed over is thus a sort of upper bound on the possible true value of action a, with
c   determining the confidence level.
    Attributes:
        C: A adjust parameter which decide how much weight it will assign to second term, the larger
        C encourage the agent to take some unknown actions.

    """
    # CODE GOES HERE
    def __init__(self, C=0.1):
        self.C= C

    def __str__(self):
        return "UCB Policy: {e}".format(e=self.C)

    def choose_action(self, agent_data):
        """
        This is where we override the method that's in the base class.

        Args:
            action_value_estimates: A numpy array containing the action-value
            action_counts: to count the number of each action happened.
            time_step: the time step up to now.
        Returns:
            action: The index of the chosen action.
        """
        action_value_estimates = agent_data["action_value_estimates"]
        action_counts = agent_data["action_counts"]
        time_step = np.sum(action_counts)
        ucb_value_estimates = np.zeros(len(action_counts))
        for i in np.arange(len(action_counts)):
            if action_counts[i]!=0:
                ucb_value_estimates[i] = action_value_estimates[i] + self.C * np.sqrt(np.log(time_step) / action_counts[i])
            else:
                ucb_value_estimates[i] = sys.float_info.max
        action = self.argmax_with_random_tiebreaker(ucb_value_estimates)
        return action



class BanditSoftmaxPolicy(BanditPolicy):
    """
    As in equation 2.9 in the text.
    """

    def __str__(self):
        return "Softmax Policy"

    def softmax(self, x):
        probabilities = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return probabilities

    def choose_action(self, agent_data):
        action_value_estimates = agent_data["action_value_estimates"]
        probabilities = self.softmax(action_value_estimates)
        action_choices = range(0,len(action_value_estimates))
        action = random.choices(action_choices, probabilities)[0]
        return action

