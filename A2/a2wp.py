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
    def __init__(self, features):
        self.features = features

    def __eq__(self, other):
        ''' Return True if self State and other State are equivalent,
        and False otherwise.'''
        return self.features == other.features
	
    def __str__(self):
        ''' Return a string representation of the current state.'''
        txt = "The current GDP is " + str(self.features[CGDP_IDX][0])
        txt += " There are " + str(self.features[FUNDS_IDX][0]) + " funds remaining"
        txt += " and " + str(TOTAL_MOVES - self.features[MOVES_IDX][0]) + " moves remaining.\n"
        return txt				

    def __hash__(self):
        ''' Return the hash of this state.'''
        return (self.__str__()).__hash__()

    def copy(self):
        ''' Return a deep copy of the current state.'''
        newS = State(self.features[:])
        return newS
		
    def can_alloc(self, actor, fund):
        ''' Return True if allocating the amount of funds in fund to actor
        does not cause a depression and there are moves remaining,
        otherwise, return False.'''

        if self.features[FUNDS_IDX][0] - fund < 0:
            return False
            
        if self.features[MOVES_IDX][0] < TOTAL_MOVES - 1:
            news = self.alloc(actor, fund)
            currentGDP = news.features[CGDP_IDX][0]
            return currentGDP >= MIN_GDP and currentGDP <= MAX_GDP and news.features[FUNDS_IDX][0] >= 0 and news.features[MOVES_IDX][0] <= 9
            
        return False

    def alloc(self, actor, fund):
        ''' Allocate the amount of funding in fund to the actor.'''
        if self.features[FUNDS_IDX][0] - fund < 0:
            return self
            
        news = self.copy()
        #featuresList = news.features
        d = {'A': self.autoStabilizer(fund),
             'M': self.monPol(fund),
             'F': self.fisPol(fund)}
        
        investment = d[actor]
        
        # Let currMoveIdx refer to the current index in the returns list.
        currMoveIdx = news.features[MOVES_IDX][0]
        
        # Put the latest investment in the returns list for future access.
        news.features[RETURNS_IDX][currMoveIdx] += investment
        
        # Get a previous investment from the returns list for updating the current GDP.
        incGDP = news.features[RETURNS_IDX][currMoveIdx]
        
        # Calculate the latest GDP. 
        newGDP = self.calcGDP(news.features[CGDP_IDX][0], incGDP)
        
        # Update the new state's features
        news.features[CGDP_IDX][0] = newGDP
        news.features[MOVES_IDX][0] += 1
        news.features[FUNDS_IDX][0] -= fund

        print('return from alloc', news.features)
        return news
            
    def calcGDP(self, currentGDP, incGDP):
        '''Calculates the new current GDP'''
        currentGDP *= (1-INIT_DECAY)
        currentGDP += incGDP
        return currentGDP

    def autoStabilizer(self, alloc):
        '''Determines the GDP increase for AS'''
        #features = self.features
        return INIT_WEIGHTS[0] * alloc * (INIT_GAMMA[0] ** self.features[MOVES_IDX][0]) 

    def monPol(self, alloc):
        '''Determine the GDP increase for MP'''
        #features = self.features
        return INIT_WEIGHTS[1] * alloc * (INIT_GAMMA[1] ** self.features[MOVES_IDX][0])

    def fisPol(self, alloc):
        '''Determine the GDP increase for FP'''
        #features = self.features
        return INIT_WEIGHTS[2] * alloc * (INIT_GAMMA[2] ** self.features[MOVES_IDX][0])

def goal_test(s):
    #return True
    print("moves: ", s.features[MOVES_IDX][0])
    return s.features[MOVES_IDX][0] == (TOTAL_MOVES - 1)

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
NUM_FEATURES = 4
TOTAL_MOVES = 10
INIT_DECAY = 0
INIT_WEIGHTS = [1.25, 1.5, 2.0]
INIT_GAMMA = [0.8, 0.9, 0.95]
#INIT_DELAY = [1, 3, 5]
# Let MAX_FUNDS be the total amount of money to fund the economy with at the
# start of the formulation. We define a constant here since the value it 
# represents is used in more than one place.
INIT_FUNDS = [250]
INIT_MOVES = [0]
INIT_RETURNS = [0] * (TOTAL_MOVES)
#INIT_RETURNS = [i for i in range(TOTAL_MOVES + INIT_DELAY[-1])]
#INIT_CGDP = [MIN_GDP*(1+INIT_DECAY)+1]
INIT_CGDP = [1250]
#INIT_MRA = ['']


# Let the following constants be the indices where their respective features can
# be accessed in the feaures list.
FUNDS_IDX = 0
MOVES_IDX = 1
RETURNS_IDX = 2
CGDP_IDX = 3
#MRA_IDX = 4

# Let INIT_FEATURES represent the features of the starting state of the economy. 
# The feature list should be of the following form:
# I_F[0] represents the funds currently allocated to the 3 components (AS, MP, FP)
# I_F[1] represents the weights funds allocated to the components have on GDP
# I_F[2] represents the amounts by which each of the components depreciates each move
# I_F[3] represents the time delay for funds allocated to the components to affect GDP
# I_F[4] represents the total funds available to allocate to the 3 components
# I_F[5] represents the total number of actions over which stave off a depression
# Note that each element in INITIAL_FUNDS is a list to simplify the copy and eq functions.
INIT_FEATURES = [[]] * NUM_FEATURES
INIT_FEATURES[FUNDS_IDX] = INIT_FUNDS
INIT_FEATURES[MOVES_IDX] = INIT_MOVES
INIT_FEATURES[RETURNS_IDX] = INIT_RETURNS
INIT_FEATURES[CGDP_IDX] = INIT_CGDP

# TODO - I made mention of representing the delay in effectiveness of each
# each of the financial polices, namely that AS is faster than MP is faster than
# FP, despite the fact that FP is more effective than MP is more effective than AP. 
# I think we should represent these efficacies as a list of values, we'll call it d. 
# So, let's say we have 10 moves, then initialize d = [0] * moves. 
# Now, if use a turn to distribute funds to AS, then the funds go to d[i + x] where
# i is the current move and x is some positive integer.
# If we distribute funds to MP, the funds go to d[i + y] where y is > x.
# And funds to FP got to d[i + z] where z > y. 
# Then, when calculating GDP on a given move i, we access the funds in d[i] as 
# part of that calculation.
# In this way we can capture the effect that yes, the funds might be more effective
# given to FP but that effect will take longer to be realized. 

CREATE_INITIAL_STATE = lambda : State(INIT_FEATURES)
#</INITIAL_STATE>

#<OPERATORS>

# We'll merge actors and funds into a list of tuples called actions.
# Let actions be of the form [(actors[0], funds[0]), (actors[0], funds[1]), ... (actors[2], funds[n])]
# where n is the length of funds. 
actors = ['A', 'M', 'F']
funds = [f for f in range(0, INIT_FUNDS[0], 20)]
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
