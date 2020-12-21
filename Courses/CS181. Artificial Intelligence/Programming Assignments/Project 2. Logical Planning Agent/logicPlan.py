# logicPlan.py
# ------------
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


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game


pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()

def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    s1 = A | B
    s2 = ~A % (~B | C)
    s3 = logic.disjoin(~A, ~B, C)

    x = logic.conjoin(s1,s2,s3)
    return x
    util.raiseNotDefined()

def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A,B,C,D = logic.Expr('A'), logic.Expr('B'), logic.Expr('C'), logic.Expr('D')
    s1 = C % (B | D)
    s2 = A >> (~B & ~D)
    s3 = ~(B & ~C) >> A
    s4 = ~D >> C

    x = logic.conjoin(s1,s2,s3,s4)
    return x
    util.raiseNotDefined()

def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WumpusAlive0, WumpusAlive1, WumpusBorn0, WumpusBorn1, WumpusKilled0 = logic.PropSymbolExpr('WumpusAlive[0]'), logic.PropSymbolExpr('WumpusAlive[1]'),\
        logic.PropSymbolExpr('WumpusBorn[0]'), logic.PropSymbolExpr('WumpusBorn[1]'), logic.PropSymbolExpr('WumpusKilled[0]')
    s1 = WumpusAlive1 % ((WumpusAlive0 & ~WumpusKilled0) | (~WumpusAlive0 & WumpusBorn0))
    s2 = ~(WumpusAlive0 & WumpusBorn0)
    s3 = WumpusBorn0
    
    x = logic.conjoin(s1, s2, s3)
    return x
    util.raiseNotDefined()

def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"

    cnf = logic.to_cnf(sentence)
    assign = logic.pycoSAT(cnf)
    return assign
    util.raiseNotDefined()

def atLeastOne(literals) :
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    x = None
    for i in literals:
        if x == None:
            x = i
        else:
            x = x | i 
    return x
    util.raiseNotDefined()


def atMostOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    new_literals = []
    for i in literals:
        for j in literals:
            if j != i:
                disjuc = logic.disjoin(~i, ~j)
                new_literals.append(disjuc)

    x = logic.conjoin(new_literals)
    return x
    util.raiseNotDefined()


def exactlyOne(literals) :
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))
    util.raiseNotDefined()


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    action_list = []
    for i in model:
        key_i = i
        value_i = model[i]
        parsed = logic.PropSymbolExpr.parseExpr(i)
        if value_i == True:
            if parsed[0] in actions:
                action_list.append(parsed)
    
    action_list = sorted(action_list, key=lambda x: int(x[1]))
    #print(action_list)
    sorted_action_list = []
    for i in action_list:
        sorted_action_list.append(i[0])
    return sorted_action_list
    util.raiseNotDefined()


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    curr_state = logic.PropSymbolExpr(pacman_str, x, y, t)
    poss_prev_states = []
    for i in [-1, 1]:
        if walls_grid[x+i][y] == False:
            prev_state = logic.PropSymbolExpr(pacman_str,x+i,y,t-1)
            prev_action = logic.PropSymbolExpr('East', t-1) if i == -1 else logic.PropSymbolExpr('West', t-1)
            next_state = logic.conjoin(prev_state, prev_action)
            poss_prev_states.append(next_state)
    for j in [-1, 1]:
        if walls_grid[x][y+j] == False:
            prev_state = logic.PropSymbolExpr(pacman_str,x,y+j,t-1)
            prev_action = logic.PropSymbolExpr('North', t-1) if j== -1 else logic.PropSymbolExpr('South', t-1)
            next_state = logic.conjoin(prev_state, prev_action)
            poss_prev_states.append(next_state)
    
    poss_prev_states = atLeastOne(poss_prev_states)
    axiom = curr_state % poss_prev_states
    return axiom
    util.raiseNotDefined()


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    "*** YOUR CODE HERE ***"
    initial_state = problem.getStartState()
    goal_state = problem.getGoalState()
    actions = ['North', 'East', 'South', 'West']

    # Only one initial state
    initial_judge = []
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if len(initial_judge) != 0:
                if (x, y) != initial_state:
                    pse = logic.PropSymbolExpr('P', x, y, 0)
                    pse = logic.Expr('~',  pse)
                    initial_judge[0] = logic.conjoin(initial_judge[0], pse)
                else:
                    initial_judge[0] = logic.conjoin(initial_judge[0], logic.PropSymbolExpr('P', x, y, 0))
            else:
                if (x, y) != initial_state:
                    pse = logic.PropSymbolExpr('P', x, y, 0)
                    pse = logic.Expr('~',  pse)
                    initial_judge.append(pse)
                else:
                    initial_judge.append(logic.PropSymbolExpr('P', x, y, 0))
    #print(initial_judge)

    action_judge = None
    succ_judge = None
    for i in range(1,50):
        # Action
        inner_action_judge = []
        for a in actions:
            inner_action_judge.append(logic.PropSymbolExpr(a, i-1))
        if action_judge == None:
            action_judge = exactlyOne(inner_action_judge)
        else:
            action_judge = logic.conjoin(action_judge, exactlyOne(inner_action_judge))
        #print(action_judge)

        # Successor
        inner_succ_judge = None
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if (x, y) not in walls.asList():
                    if inner_succ_judge != None:
                        inner_succ_judge = logic.conjoin(inner_succ_judge, pacmanSuccessorStateAxioms(x,y,i,walls))
                    else:
                        inner_succ_judge = pacmanSuccessorStateAxioms(x,y,i,walls)
        if succ_judge != None:
            succ_judge = logic.conjoin(succ_judge, inner_succ_judge)
        else: 
            succ_judge = inner_succ_judge
        #print(succ_judge)

        # Goal
        is_goal_state = logic.PropSymbolExpr('P', goal_state[0], goal_state[1], i+1)
        goal_judge = logic.conjoin(is_goal_state, pacmanSuccessorStateAxioms(goal_state[0], goal_state[1], i+1, walls))
        
        # Final expr
        final_expr = logic.conjoin(initial_judge[0], action_judge, succ_judge, goal_judge)
        result = findModel(final_expr)
        if result != False:
            return extractActionSequence(result, actions)

    util.raiseNotDefined()


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    initial_state = problem.getStartState()
    actions = ['North', 'East', 'South', 'West']
    pac_loaction = initial_state[0]
    food_loaction = initial_state[1].asList()
    # Only one initial state
    initial_judge = []
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if len(initial_judge) != 0:
                if (x, y) != pac_loaction:
                    pse = logic.PropSymbolExpr('P', x, y, 0)
                    pse = logic.Expr('~',  pse)
                    initial_judge[0] = logic.conjoin(initial_judge[0], pse)
                else:
                    initial_judge[0] = logic.conjoin(initial_judge[0], logic.PropSymbolExpr('P', x, y, 0))
            else:
                if (x, y) != initial_state:
                    pse = logic.PropSymbolExpr('P', x, y, 0)
                    pse = logic.Expr('~',  pse)
                    initial_judge.append(pse)
                else:
                    initial_judge.append(logic.PropSymbolExpr('P', x, y, 0))
    #print(initial_judge)

    action_judge = None
    succ_judge = None
    for i in range(1,50):
        # Action
        inner_action_judge = []
        for a in actions:
            inner_action_judge.append(logic.PropSymbolExpr(a, i-1))
        if action_judge == None:
            action_judge = exactlyOne(inner_action_judge)
        else:
            action_judge = logic.conjoin(action_judge, exactlyOne(inner_action_judge))

        # Successor
        inner_succ_judge = None
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                if (x, y) not in walls.asList():
                    if inner_succ_judge != None:
                        inner_succ_judge = logic.conjoin(inner_succ_judge, pacmanSuccessorStateAxioms(x,y,i,walls))
                    else:
                        inner_succ_judge = pacmanSuccessorStateAxioms(x,y,i,walls)
        if succ_judge != None:
            succ_judge = logic.conjoin(succ_judge, inner_succ_judge)
        else: 
            succ_judge = inner_succ_judge
        #print(succ_judge)

        # Food
        inner_food_judge = None
        for each_food in food_loaction:
            d_inner_food_judge = None
            for t in range(i+1):
                if d_inner_food_judge != None:
                    d_inner_food_judge = logic.disjoin(d_inner_food_judge, logic.PropSymbolExpr('P', each_food[0], each_food[1], t))
                else:
                    d_inner_food_judge = logic.PropSymbolExpr('P', each_food[0], each_food[1], t)
            if inner_food_judge != None:
                inner_food_judge = logic.conjoin(inner_food_judge, d_inner_food_judge)
            else:
                inner_food_judge = d_inner_food_judge

        # Final expr
        final_expr = logic.conjoin(initial_judge[0], action_judge, succ_judge, inner_food_judge)
        result = findModel(final_expr)
        if result != False:
            return extractActionSequence(result, actions)

    util.raiseNotDefined()



# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
    