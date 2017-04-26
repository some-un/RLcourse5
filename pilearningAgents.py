# PILearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import numpy as np
import random,util,math

class PILearningAgent(ReinforcementAgent):
    """
      PI-Learning Agent

      Functions you should fill in:
        - getPiValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.valuefcn = util.Counter()
        self.saprob = util.Counter() # state-action pair encountering "probability"
        self.feats = util.Counter()
        self.theta = util.Counter()
        self.tcounter = 0
        
    def getPiValue(self, state, action): #THIS RETURNS POLICY DISTRIBUTIONS NOW !!
        """
          Returns Pi(state,action)
        """
        return self.saprob[(state,action)] #your pi value

    def getAction(self, state):
        """
          Compute the action to take in the current state.
        """
        piValuesList = util.Counter()
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            # terminal state case, no legal actions
            return None
        for a in legalActions:
            piValuesList[a] = self.getPiValue(state,a) # self.saprob[(state,a)]
        actionToBeTaken = None
        if self.epsilon > 0:
            # training phase
            ourp = None
            if sum(piValuesList.values()) is 1:
                ourp = piValuesList.values()
            # if not, print "sum(piValuesList.values()) is: ", sum(piValuesList.values()), ", setting p to None for uniform distribution."
            actionToBeTaken = np.random.choice(legalActions, p = ourp)
        elif self.epsilon == 0:
            # game phase
            actionToBeTaken = piValuesList.argMax()
        #if actionToBeTaken is None or actionToBeTaken == []:
        #    actionToBeTaken = legalActions[0]
        return actionToBeTaken
    
    def update(self, state, action, nextState, reward):
        """
        update function
        """
        delta = reward + self.discount * self.valuefcn[nextState] - self.valuefcn[state]
        #
        legalActions = self.getLegalActions(state)
        self.feats[(state,action)] = 1 if self.getPiValue(state,action) > 0 else 0
        scoreVal = 0
        for a in legalActions:
            scoreVal += self.feats[(state,a)] * self.getPiValue(state,a)
        self.theta[(state,action)] += self.alpha * delta * (self.feats[(state,action)] - scoreVal)
        #print "delta: ", delta, " | scoreVector: ", scoreVal  
        # updating policy
        denominator = 0 #= np.float64(0)
        for a in legalActions:
            featTimesParam = self.feats[(state,a)] * self.theta[(state,a)]
            denominator += math.exp(featTimesParam)
        if denominator == 0:
            self.saprob[(state,action)] = 0
        else:
            self.saprob[(state,action)] = math.exp(self.feats[(state,action)] * self.theta[(state,action)]) / denominator
        #
        # introducing fix to task1, reseting feature value to 0 after theta update
        # as it's not meant to be an eligibility trace in here!
        #
        # reseting feature value
        self.feats[(state,action)] = 0
        #
        self.valuefcn[state] += self.alpha * delta
        #

class PacmanPIAgent(PILearningAgent):
    "Exactly the same as PILearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanPILearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        PILearningAgent.__init__(self, **args)

    def getAction(self, state):
        """

        """
        action = PILearningAgent.getAction(self,state)
        self.doAction(state,action)

        return action

class ApproximatePIAgent(PacmanPIAgent):
    """

    """
    # Seb note: Pacman has its own getPiValue function
    
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanPIAgent.__init__(self, **args)
        
        #Note: states are not known apriori, empty dictionaries
        self.Pi        = util.Counter() # Empty        policy weights
        self.V         = {} # Empty        value function weights
        self.ET_glogPi = {} # Empty Eligibility trace for gradient of log Pi
        self.ET_V      = util.Counter() # Empty Eligibility trace for V
        self.lambdaVal = 0.75

    def getPiValue(self, state, action): #REPORTS POLICY DISTRIBUTION !!
        """

        """
        return self.saprob[(state,action)]
    
    def getVValue(self, state):
        return self.valuefcn[state]
            
    def update(self, state, action, nextState, reward):
        """
        """
        delta = reward + self.discount * self.getVValue(nextState) - self.getVValue(state)
        #
        legalActions = self.getLegalActions(state)
        expValUnderPi = 0
        for a in legalActions:
            for f in self.featExtractor.getFeatures(state,a).values():
                expValUnderPi += f * self.getPiValue(state, a)
        self.ET_V[state] = self.lambdaVal * self.ET_V[state] + expValUnderPi
        #
        # Eligibility trace for PI values        
        theFeatures = self.featExtractor.getFeatures(state,action).values()
        #
        self.Pi[(state,action)] = self.lambdaVal * self.Pi[(state,action)] + (theFeatures - expValUnderPi)
        #
        self.theta[(state,action)] += self.alpha * delta * self.Pi[(state,action)]
        self.valuefcn[state] += self.alpha * delta * self.ET_V[state]
    
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanPIAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            
            print 'Policy weights: ' #your weights
