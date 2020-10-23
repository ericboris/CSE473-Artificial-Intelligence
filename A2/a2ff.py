'''a2ff.py
by Terry Nguyen and John Nathan Smith
UWNetIDs: ternguyen5,  jnsmith98
Student numbers: 1820381, and 1742903

Assignment 2, in CSE 473, Autumn 2020.

Part A
 
This file contains our problem formulation for the problem of
the Farmer, Fox, Chicken, and Grain.
'''

# Put your formulation of the Farmer-Fox-Chicken-and-Grain problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.

#<METADATA>
SOLUZION_VERSION = "2.0"
PROBLEM_NAME = "Farmer-Fox-Chicken-Grain"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['T. Nguyen', 'J.N. Smith']
PROBLEM_CREATION_DATE = "22-OCT-2020"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"Farmer-Fox-Chicken-Grain"</b> problem is a traditional puzzle
(add description).

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
  return "You did it." # CHANGE THE MESSAGE

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





