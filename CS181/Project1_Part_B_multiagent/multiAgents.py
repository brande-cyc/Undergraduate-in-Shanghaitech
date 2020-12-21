# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        optimal_value, optimal_action = self.minmax(gameState, 0)
        return optimal_action

        util.raiseNotDefined()

    def max_value(self, gameState, depth):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        max_score = -10000000    # Initialized with a small number
        optimal_action = None   
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the min value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.minmax(succ, depth + 1)
                if succ_value > max_score:
                    max_score = succ_value
                    optimal_action = action
        
        return max_score, optimal_action

    def min_value(self, gameState, depth):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        min_score = 10000000   # Initialized with a large number
        optimal_action = None 
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the min value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.minmax(succ, depth + 1)
                if succ_value < min_score:
                    min_score = succ_value
                    optimal_action = action
        
        return min_score, optimal_action

    def minmax(self, gameState, depth):
        # If we come to the leave of the minmax tree or win or lose, terminate
        if depth == self.depth * gameState.getNumAgents():
            value_leave_node = self.evaluationFunction(gameState)
            return value_leave_node, None
        # Else if we come to the end of the game
        elif gameState.isWin() or gameState.isLose():
            value_end_node = self.evaluationFunction(gameState)
            return value_end_node, None
        # if it is a pacman, then we use maximizer
        elif depth % gameState.getNumAgents() == 0:
            value_pacman, action_pacman = self.max_value(gameState, depth)
            return value_pacman, action_pacman
        # if it is a ghost, then we use minimizer
        elif depth % gameState.getNumAgents() != 0:
            value_ghost, action_ghost = self.min_value(gameState, depth)
            return value_ghost, action_ghost


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, optimal_action = self.AlphaBeta(gameState, 0, alpha = -1000000, beta = 1000000)
        return optimal_action
        util.raiseNotDefined()

    def max_value(self, gameState, depth, alpha, beta):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        max_score = -10000000    # Initialized with a small number
        optimal_action = None
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the min value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.AlphaBeta(succ, depth + 1, alpha, beta)
                if succ_value > max_score:
                    max_score = succ_value
                    optimal_action = action
                # if max_score > beta, then we can prune
                if max_score > beta:
                    return max_score, optimal_action
                # update alpha to the larger one
                alpha = max(alpha, max_score)
        
        return max_score, optimal_action

    def min_value(self, gameState, depth, alpha, beta):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        min_score = 10000000   # Initialized with a large number
        optimal_action = None
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the min value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.AlphaBeta(succ, depth + 1, alpha, beta)
                if succ_value < min_score:
                    min_score = succ_value
                    optimal_action = action
                # if min_score < alpha, then we can prune
                if min_score < alpha:
                    return min_score, optimal_action, 
                # update beta to the smaller one
                beta = min(beta, min_score)
            
            return min_score, optimal_action


    def AlphaBeta(self, gameState, depth, alpha, beta):
        # If we come to the leave of the minmax tree or win or lose, terminate
        if depth == self.depth * gameState.getNumAgents():
            value_leave_node = self.evaluationFunction(gameState)
            return value_leave_node, None
        # Else if we come to the end of the game
        elif gameState.isWin() or gameState.isLose():
            value_end_node = self.evaluationFunction(gameState)
            return value_end_node, None
        # if it is a pacman, then we use maximizer
        elif depth % gameState.getNumAgents() == 0:
            value_pacman, action_pacman = self.max_value(gameState, depth, alpha, beta)
            return value_pacman, action_pacman
        # if it is a ghost, then we use minimizer
        elif depth % gameState.getNumAgents() != 0:
            value_ghost, action_ghost = self.min_value(gameState, depth, alpha, beta)
            return value_ghost, action_ghost

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, optimal_action = self.expectimax(gameState, 0)
        return optimal_action
        util.raiseNotDefined()
    
    def max_value(self, gameState, depth):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        max_score = -10000000    # Initialized with a small number
        optimal_action = None   
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the max value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.expectimax(succ, depth + 1)
                if succ_value > max_score:
                    max_score = succ_value
                    optimal_action = action
        
        return max_score, optimal_action

    def exp_value(self, gameState, depth):
        agent_index = depth % gameState.getNumAgents()  # index of the current agent
        actions = gameState.getLegalActions(agent_index)    # get valid actions
        expecti_value = 0
        weight = 1./len(actions)
        # if it is a leave node
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        else:
            # if not, we need to find the expecti value
            for i, action in enumerate(actions):
                succ = gameState.generateSuccessor(agent_index, action)
                succ_value, _ = self.expectimax(succ, depth + 1)
                expecti_value += weight * succ_value
        
        return expecti_value, None
    
    def expectimax(self, gameState, depth):
        # If we come to the leave of the minmax tree or win or lose, terminate
        if depth == self.depth * gameState.getNumAgents():
            value_leave_node = self.evaluationFunction(gameState)
            return value_leave_node, None
        # Else if we come to the end of the game
        elif gameState.isWin() or gameState.isLose():
            value_end_node = self.evaluationFunction(gameState)
            return value_end_node, None
        # if it is a pacman, then we use maximizer
        elif depth % gameState.getNumAgents() == 0:
            value_pacman, action_pacman = self.max_value(gameState, depth)
            return value_pacman, action_pacman
        # if it is a ghost, then we use minimizer
        elif depth % gameState.getNumAgents() != 0:
            value_ghost, action_ghost = self.exp_value(gameState, depth)
            return value_ghost, action_ghost


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_state = currentGameState.getPacmanState()
    pacman_position = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    ghost_scared_times = [] # Scared condition of ghosts
    for ghost_state in ghost_states:
        ghost_scared_times.append(ghost_state.scaredTimer)
    capsule_position = currentGameState.getCapsules()   # Capsules list
    food = currentGameState.getFood()
    foods = food.asList()
    score = 0   # Final score

    # Find the nearest ghost. And we consider the scared time as additional bonus
    nearest_ghost_distance = 10000000
    for i, ghost in enumerate(ghost_states):
        ghost_position = ghost.getPosition()
        scare_time = ghost_scared_times[i]
        nearest_ghost_distance = min(nearest_ghost_distance, util.manhattanDistance(ghost_position, pacman_position) - scare_time + 1)
    score -= nearest_ghost_distance * 0.5

    # Find the nearest food
    nearest_food_distance = 10000000
    # if no food, distance = 0
    if len(foods) == 0:
        nearest_food_distance = 0
    else:
        for i, food in enumerate(foods):
            nearest_food_distance = min(nearest_food_distance, util.manhattanDistance(food, pacman_position)) 
    score -= nearest_food_distance
    
    # if win, score += 1000
    if currentGameState.isWin():
        score += 1000
    # if pacman find capsule, score += scareTime
    if pacman_position in capsule_position:
        score += 40
    
    return score + currentGameState.getScore()
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
