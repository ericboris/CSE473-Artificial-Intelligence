'''a2wp.py
by Eric Boris and TODO
UWNetIDs: eboris91, TODO
Student numbers: 1976637, and TODO

Assignment 2, in CSE 473, Autumn 2020.
PART B
 
This file contains our problem formulation for preventing a depression in the US.
'''

# Put your formulation of your chosen wicked problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.

#<METADATA>
SOLUZION_VERSION = "2.0"
PROBLEM_NAME = "Avoiding a Severe Economic Depression in the US"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['E. Boris', 'TODO']
PROBLEM_CREATION_DATE = "22-Oct-2020"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"Avoiding a Severe Economic Depression in the US"</b>
is a wicked problem in which a country (the US) starts off in an economic state determined by:
1. a GDP with funds distributed between three components: Auto Stablizers (As), Monetary Policies (Mp),
and Fiscal Policies (Fp), 2. a predetermined amount of time steps over which to maintain economic 
stability, and 3. a total amount of wealth to distribute into the economy. The object is to distribute
some quantity of funding 0 <= f <= totalFunds to one of the three components of GDP (As, Mp, Fp) on any given
action with the intent of maintaining the GDP above a minimum threshold represented by the starting GDP.
This is to say, that the GDP may never drop below the starting point. The health of the economy must be
maintained above this threshold for the predetermined amount of time for success to be achieved.      
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>

class State():
    def __init__(self, gdp, funds, move, returns):
        self.gdp = gdp
        self.funds = funds
        self.move = move
        self.returns = returns

    def __eq__(self, other):
        ''' Return True if self State and other State are equivalent,
        and False otherwise.'''
        return self.gdp == other.gdp and self.funds == other.funds and self.move == other.move and self.returns == other.returns
	
    def __str__(self):
        ''' Return a string representation of the current state.'''
        txt = "The current GDP is " + str(int(self.gdp))
        txt += " there are $" + str(int(self.funds)) + " left in funds"
        txt += " and " + str(TOTAL_MOVES - self.move) + " moves remaining.\n"
        return txt				

    def __hash__(self):
        ''' Return the hash of this state.'''
        return (self.__str__()).__hash__()

    def copy(self):
        ''' Return a deep copy of the current state.'''
        return State(self.gdp, self.funds, self.move, self.returns[:])
		
    def can_alloc(self, actor, fund):
        ''' Return True if allocating the amount of funds in fund to actor
        does not cause a depression and there are moves remaining,
        otherwise, return False.'''

        if self.funds - fund < 0 or self.move >= TOTAL_MOVES:
            return False
            
        news = self.alloc(actor, fund)
        return news.gdp >= MIN_GDP and news.gdp <= MAX_GDP

    def alloc(self, actor, fund):
        ''' Allocate the amount of funding in fund to the actor.'''           
        news = self.copy()

        d = {'A': (calcPolicyReturn(fund, self.move, WEIGHT[0], GAMMA[0]), DELAY[0]),
             'M': (calcPolicyReturn(fund, self.move, WEIGHT[1], GAMMA[1]), DELAY[1]),
             'F': (calcPolicyReturn(fund, self.move, WEIGHT[2], GAMMA[2]), DELAY[2])}
        
        investment, delay = d[actor]
        
        # Get a previous investment from the returns list for updating the current GDP.
        incGDP = news.returns[self.move]
        
        news.gdp = calcGDP(self.gdp, incGDP)
        news.move += 1
        news.funds -= fund
        news.returns[self.move + delay] += investment

        print('return from alloc', news.returns)
        return news

def calcGDP(currentGDP, incGDP):
    '''Calculates the new current GDP'''
    currentGDP *= (1 - DECAY)
    currentGDP += incGDP
    return currentGDP

def calcPolicyReturn(alloc, move, weight, gamma):
    return weight * alloc * (gamma ** move)

def goal_test(s):
    return s.move >= TOTAL_MOVES - 1 and s.gdp >= MIN_GDP and s.gdp <= MAX_GDP

def goal_message(s):
    return "You prevented a severe depression!"

def get_name(actor, fund):
    actorMap = {'A': 'Automatic Stabilizers', 'M': 'Monetary Policy', 'F': 'Fiscal Policy'}
    return 'Allocate ' + str(fund) + ' in funding to ' + actorMap[actor] + '.'

class Operator:
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        return self.precond(s)

    def apply(self, s):
        return self.state_transf(s)
#</COMMON_CODE>

#<INITIAL_STATE>
MIN_GDP = 1000
MAX_GDP = 2000
TOTAL_MOVES = 10
DECAY = 0.025
WEIGHT = [1.25, 1.5, 2.0]
GAMMA = [0.8, 0.9, 0.95]
DELAY = [1, 3, 5]

INIT_GDP = 1250
INIT_FUNDS = 100
INIT_MOVES = 0
INIT_RETURNS = [0] * (TOTAL_MOVES + DELAY[-1])


CREATE_INITIAL_STATE = lambda : State(INIT_GDP, INIT_FUNDS, INIT_MOVES, INIT_RETURNS)
#</INITIAL_STATE>

#<OPERATORS>

# We'll merge actors and funds into a list of tuples called actions.
# Let actions be of the form [(actors[0], funds[0]), (actors[0], funds[1]), ... (actors[2], funds[n])]
# where n is the length of funds. 
actors = ['A', 'M', 'F']
funds = [f for f in range(0, INIT_FUNDS+1, 50)]
actions = [(a, f) for a in actors for f in funds]

OPERATORS = [Operator(get_name(actor, fund),
                        lambda state, a=actor, f=fund : state.can_alloc(a, f),
                        lambda state, a=actor, f=fund : state.alloc(a, f))
                        for actor, fund in actions]

#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>
