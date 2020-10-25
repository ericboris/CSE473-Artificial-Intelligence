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
SOLUZION_VERSION = '2.0'
PROBLEM_NAME = 'Farmer-Fox-Chicken-Grain'
PROBLEM_VERSION = '1.0'
PROBLEM_AUTHORS = ['T. Nguyen', 'J.N. Smith']
PROBLEM_CREATION_DATE = '22-OCT-2020'

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>'Farmer-Fox-Chicken-Grain'</b> problem is a traditional puzzle consisting of a Farmer, fox, chicken, and grain one bank of a river. The goal of the puzzle is to move all four to the other side, however, the Farmer can only take either the fox, chicken, or grain across at any one time and the Farmer cannot leave the fox alone with the chicken nor leave the chicken alone with the grain. In what order must the Farmer move each across the river?
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>

class State():
    def __init__(self, d):
        self.d = d
        self.illegals = [{'f', 'c'}, {'c', 'g'}]

    def __eq__(self, other):
        for val in ['L', 'R']:
            if self.d[val] != other.d[val]: return False
        return True
    
    '''def __str__(self):
        #TODO:'''

    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        news = State({})
        for side in ['L', 'R']:
            news.d[side] = set([e for e in self.d[side]])
        return news

    def can_move(self, source, dest, actors):
        news = self.move(source, dest, actors)
        return news.d['L'] not in self.illegals and news.d['R'] not in self.illegals

    def move(self, source, dest, actors):
        news = self.copy()
        #print('paramters: s:', source, ' d:', dest, ' a:', actors)
        for actor in actors:
            if actor in news.d[source]:
                news.d[source].remove(actor)
                news.d[dest].add(actor)
        return news
        
def goal_test(s):
    goalState = State({'L': set(), 'R': {'F', 'f', 'c', 'g'}})  
    return s.d == goalState.d
    
def goal_message(s):
  return 'You did great!' # CHANGE THE MESSAGE

def str_format(src, dst, actors):
    return 'Move ' + str(actors) + ' from ' + src + ' ' + dst 


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
INITIAL_DICT = {'L': {'F', 'f', 'g', 'c'}, 'R': set()}
CREATE_INITIAL_STATE = lambda : State(INITIAL_DICT) # TODO FIX THIS
#</INITIAL_STATE>

#<OPERATORS>
actions = [('L', 'R', ['F']),
           ('L', 'R', ['F', 'f']),
           ('L', 'R', ['F', 'c']), 
           ('L', 'R', ['F', 'g']), 
           ('R', 'L', ['F']),
           ('R', 'L', ['F', 'f']),
           ('R', 'L', ['F', 'c']),  
           ('R', 'L', ['F', 'g'])]

OPERATORS = [Operator(lambda s1 = src, d1=dst, a1=actors: str_format(s1,d1,a1),
                      lambda s, s1 = src, d1=dst, a1=actors: s.can_move(s1, d1, a1),
                      lambda s, s1 = src, d1=dst, a1=actors: s.move(s1, d1, a1) )
            for (src, dst, actors) in actions] 
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>





