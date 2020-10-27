'''a2wp.py
by Eric Boris and TODO
UWNetIDs: eboris91,  jnsmith98
Student numbers: 1976637, and 1742903

Assignment 2, in CSE 473, Autumn 2020.
PART B
 
This file contains our problem formulation for preventing a depression in the US.
'''

# Put your formulation of your chosen wicked problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.

#<METADATA>
SOLUZION_VERSION = "2.0"
PROBLEM_NAME = "Finding a COVID-19 Vaccine"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['E. Boris', 'TODO']
PROBLEM_CREATION_DATE = "22-Oct-2020"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"Avoiding a Severe Economic Depression in the US"</b> problem is
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
  pass

def goal_test(s):
  pass

def goal_message(s):
  return "You prevented a severe depression!" #TODO incorporate the phrase 'depression for x number of days/weeks/months!'

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
CREATE_INITIAL_STATE = lambda : None # FIX THIS
#</INITIAL_STATE>

#<OPERATORS>

# OPERATORS =     # FIX THIS

#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>





