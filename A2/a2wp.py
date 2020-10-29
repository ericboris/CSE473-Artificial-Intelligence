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
SOLUZION_VERSION = '2.0'
PROBLEM_NAME = 'Avoiding a Severe Economic Depression in the US'
PROBLEM_VERSION = '1.0'
PROBLEM_AUTHORS = ['E. Boris', 'R. Ram']
PROBLEM_CREATION_DATE = '22-Oct-2020'

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>'Avoiding a Severe Economic Depression in the US'</b>
is a wicked problem in which the country begins with certain GDP, amount of funds, and number of months
over which to allocate those funds into three investment strategies: Automatic Stablizers (As), 
Monetary Policies (Mp), and Fiscal Policies. Each investment strategy has pros and cons, for example
Automatic Stabilizes are quick to produce net positive returns on GDP but that effect is relatively small, 
Fiscal Policies however are the opposite, they take a longer time to produce returns but their effect
is much greater. Each strategy also has a gamma associated with it, that is, a diminishing return on its
effects the later that it is used. The object is to maintain a stable GDP over the period of months, in other
words, a GDP too high or too low will trigger a depression. Success is achieved when the correct combination
of policy investments is made such that no depression occurs during this timeframe.
'''
#</METADATA>

#<COMMON_DATA>
# The following constants in COMMON_DATA represent
# the parameters of the problem formulation and
# they can and should be varied to test out different
# simulations. 
# Their current settings are not intended to reflect
# their real world counterparts as much as to provide an
# interesting and computation-time reasonable simulation.

# Let MIN and MAX GDP represent the lower and 
# upper bounds within which to maintain the GDP.
MIN_GDP = 1000
MAX_GDP = 2000

# Let TOTAL_MONTHS represent the total number of 
# months to prevent a depression over.
TOTAL_MONTHS = 12

# Let INFLATION represent the rate of depreciation of 
# GDP due to inflation. 
INFLATION = 0.04

# Let WEIGHT represent the multiplicative returns on the
# 3 investment strategies: as, mp, and fp, respectively.
WEIGHT = [1.5, 2.3, 3.7]

# Let GAMMA represent the diminishing effectiveness
# of the 3 investment strategies per month step.
GAMMA = [0.5, 0.9, 0.7]

# Let DELAY represent the delay on investment returns
# on an investment strategy in months.
DELAY = [1, 3, 8]
#</COMMON_DATA>

#<COMMON_CODE>

class State():
    ''' Let a state represent the state of the current GDP.'''
    def __init__(self, gdp, funds, month, returns):
        self.gdp = gdp
        self.funds = funds
        self.month = month
        self.returns = returns

    def __eq__(self, other):
        ''' Return True if self State and other State are equivalent,
        and False otherwise.'''
        s = [self.gdp, self.funds, self.month, self.returns]
        o = [other.gdp, other.funds, other.month, other.returns]
        return all([p == q for p, q in zip(s, o)])
	
    def __str__(self):
        ''' Return a string representation of the current state.'''
        txt = 'The current GDP is $' + str(int(self.gdp))
        txt += ', there are $' + str(int(self.funds)) + ' left in funds'
        txt += ' and there are ' + str(TOTAL_MONTHS - self.month) + ' months remaining.\n'
        return txt				

    def __hash__(self):
        ''' Return the hash of this state.'''
        return (self.__str__()).__hash__()

    def copy(self):
        ''' Return a deep copy of the current state.'''
        return State(self.gdp, self.funds, self.month, self.returns[:])
		
    def can_alloc(self, policy, fund):
        ''' Return True if allocating the amount of funds in fund to policy
        does not cause a depression and there are months remaining,
        otherwise, return False.'''

        # Make sure we're not out of bounds of the problem.
        if self.funds - fund < 0 or self.month >= TOTAL_MONTHS:
            return False
            
        # Otherwise, apply the policy changes and check that 
        # the results don't through the simulation out of bounds.
        newS = self.alloc(policy, fund)
        return newS.gdp >= MIN_GDP and newS.gdp <= MAX_GDP

    def alloc(self, policy, fund):
        ''' Allocate the amount of funding in fund to the policy.'''           
        newS = self.copy()

        # Let d map an investment policy to it's return in investment as
        # a number of months until that return is realized.
        d = {'as': (policyInvestment(fund, self.month, WEIGHT[0], GAMMA[0]), DELAY[0]),
             'mp': (policyInvestment(fund, self.month, WEIGHT[1], GAMMA[1]), DELAY[1]),
             'fp': (policyInvestment(fund, self.month, WEIGHT[2], GAMMA[2]), DELAY[2])}
        
        investment, delay = d[policy]
        
        # Reap the previously invested returns.
        policyReturn = newS.returns[self.month]
        
        # Reflect the allocation of funding to a particular
        # investment policy on the new state.
        newS.gdp = newGDP(self.gdp, policyReturn)
        newS.month += 1
        newS.funds -= fund
        newS.returns[self.month + delay] += investment

        return newS

def newGDP(gdp, policyReturn):
    ''' Return the new GDP from the current GDP and the policy investment returns.'''
    return (gdp * (1 - INFLATION)) + policyReturn

def policyInvestment(alloc, month, weight, gamma):
    ''' Return the amount in returns of investing in a particular investment policy.'''
    return weight * alloc * (gamma ** month)

def goal_test(s):
    ''' Return True if a goal state is encountered and False otherwise.'''
    return s.month >= TOTAL_MONTHS and s.gdp >= MIN_GDP and s.gdp <= MAX_GDP

def goal_message(s):
    ''' Return a message for reaching a goal state. '''
    return 'You prevented a severe depression!'

def get_name(policy, fund):
    ''' Return a description of the action of funding an investment policy.'''
    policyMap = {'as': 'Automatic Stabilizers', 'mp': 'Monetary Policy', 'fp': 'Fiscal Policy'}
    return 'Allocate ' + str(fund) + ' in funding to ' + policyMap[policy] + '.'

class Operator:
    ''' Let an Operator represent the current action made on a state.'''
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
# Let the following constants represent the values
# for the initial state object.
INIT_GDP = 1250
INIT_FUNDS = 500
INIT_MONTHS = 0
INIT_RETURNS = [0] * (TOTAL_MONTHS + DELAY[-1])

CREATE_INITIAL_STATE = lambda : State(INIT_GDP, INIT_FUNDS, INIT_MONTHS, INIT_RETURNS)
#</INITIAL_STATE>

#<OPERATORS>

# We'll merge policies and funds into a list of tuples called actions.
# Let actions be of the form [(policies[0], funds[0]), (policies[0], funds[1]), ... (policies[2], funds[n])]
# where n is the length of funds. 
policies = ['as', 'mp', 'mp']
funds = [f for f in range(0, INIT_FUNDS+1, 100)]
actions = [(p, f) for p in policies for f in funds]

OPERATORS = [Operator(get_name(policy, fund),
                        lambda state, p=policy, f=fund : state.can_alloc(p, f),
                        lambda state, p=policy, f=fund : state.alloc(p, f))
                        for policy, fund in actions]

#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>
