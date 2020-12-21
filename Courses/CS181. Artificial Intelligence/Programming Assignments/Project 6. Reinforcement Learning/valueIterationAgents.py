# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
from functools import reduce
import copy

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"  
        AllStates = self.mdp.getStates()
        for i in range(self.iterations):
            UpdatedValue = copy.deepcopy(self.values)
            for s in AllStates:
                if self.mdp.isTerminal(s):
                    continue
                else:
                    AllActions = self.mdp.getPossibleActions(s)
                    UpdatedValue[s] = reduce(lambda x, y: max(x,y), [self.getQValue(s,a) for a in AllActions])
            self.values = UpdatedValue

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ModelInfo = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for SuccessorState, TransitionProb in ModelInfo:
            reward = self.mdp.getReward(state,action,SuccessorState)
            QValue += TransitionProb * (reward + self.discount * self.values[SuccessorState])
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        PossibleActions = self.mdp.getPossibleActions(state)
        ActionsValue = util.Counter()
        for a in PossibleActions:
            ActionsValue[a] = self.getQValue(state, a)
        
        BestAction = ActionsValue.argMax()
        return BestAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        AllStates = self.mdp.getStates()
        NumStates = len(AllStates)
        for i in range(self.iterations):
            s = AllStates[i % NumStates]
            if self.mdp.isTerminal(s):
                continue
            else:
                AllActions = self.mdp.getPossibleActions(s)
                self.values[s] = reduce(lambda x, y: max(x,y), [self.getQValue(s,a) for a in AllActions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        AllStates = self.mdp.getStates()
        # Find predecessors
        Predecessors = {}
        for s in AllStates:
            Predecessors[s] = set() 
        for s in AllStates:
            if self.mdp.isTerminal(s):
                continue
            else:
                for a in self.mdp.getPossibleActions(s):
                    for successor, TransProb in self.mdp.getTransitionStatesAndProbs(s, a):
                        if TransProb != 0:
                            Predecessors[successor].add(s)
        
        # Priority queue
        pqueue = util.PriorityQueue()
        for s in AllStates:
            if not self.mdp.isTerminal(s):
                OptimalValue = reduce(lambda x,y: max(x,y), [self.getQValue(s,a) for a in self.mdp.getPossibleActions(s)])
                diff = abs(OptimalValue - self.values[s])
                pqueue.push(s, -diff)

        for _ in range(self.iterations):
            if pqueue.isEmpty():
                break
            PopState = pqueue.pop()
            if not self.mdp.isTerminal(PopState):
                AllActions = self.mdp.getPossibleActions(PopState)
                self.values[PopState] = reduce(lambda x,y: max(x, y), [self.getQValue(PopState,a) for a in AllActions])

            for p in Predecessors[PopState]:
                if not self.mdp.isTerminal(p):
                    OptimalValue = reduce(lambda x,y: max(x,y), [self.getQValue(p,a) for a in self.mdp.getPossibleActions(p)])
                    diff = abs(OptimalValue - self.values[p])
                    if diff > self.theta:
                        pqueue.update(p, -diff)