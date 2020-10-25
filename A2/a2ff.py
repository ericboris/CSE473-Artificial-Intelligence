'''a2ff.py
by Eric Boris and Rahul Ram
UWNetIDs: eboris91,  TODO
Student numbers: 1976637, and TODO
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
PROBLEM_AUTHORS = ['E. Boris', 'R. Ram']
PROBLEM_CREATION_DATE = '22-OCT-2020'

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>'Farmer-Fox-Chicken-Grain'</b> problem is a traditional puzzle consisting of a Farmer, fox, chicken, and grain one bank of a river. The goal of the puzzle is to move all four to the other b, however, the Farmer can only take either the fox, chicken, or grain across at any one time and the Farmer cannot leave the fox alone with the chicken nor leave the chicken alone with the grain. In what order must the Farmer move each across the river?
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

# TODO - comment code

#<COMMON_CODE>
class State():    
    def __init__(self, banks):
        self.banks = banks

    def __eq__(self, other):
        return all([self.banks[b] == other.banks[b] for b in ('L', 'R')])
    
    def __str__(self):
        actorMap = {'F': 'Farmer', 'f': 'fox', 'c': 'chicken', 'g': 'grain'}
        if self.banks['L']:
            leftBank = 'The '
            leftBank += ' and '.join([actorMap[b] for b in self.banks['L']])
            if len(self.banks['L']) == 1:
                leftBank += ' is'
            else:
                leftBank += ' are'
        else: 
            leftBank = 'No one is'
            
        if self.banks['R']:
            rightBank = ' the '
            rightBank += ' and '.join([actorMap[b] for b in self.banks['R']])
            if len(self.banks['R']) == 1:
                rightBank += ' is'
            else:
                rightBank += ' are'
        else:
            rightBank = ' no one is'
            
        return leftBank + ' on the left bank and' + rightBank + ' on the right bank.\n'
        
    def __hash__(self):
        return (self.__str__()).__hash__()

    def copy(self):
        newS = State({})
        
        for b in ('L', 'R'):
            newS.banks[b] = set([actor for actor in self.banks[b]])
            
        return newS

    def can_move(self, actors, src, dst):
        newS = self.move(actors, src, dst)
        illegalStates = ({'f', 'c'}, {'c', 'g'}, {'f', 'c', 'g'})
        return all([newS.banks[b] not in illegalStates for b in ('L', 'R')])

    def move(self, actors, src, dst):
        newS = self.copy()

        # Only move actors from src to dst if all the actors
        # are currently on the src bank.
        if all([a in newS.banks[src] for a in actors]):
            for actor in actors:
                newS.banks[src].remove(actor)
                newS.banks[dst].add(actor)
                
        return newS
        
        
def goal_test(s):
    goal = State({'L': set(), 'R': {'F', 'f', 'c', 'g'}})  
    return s.banks == goal.banks
    
def goal_message(s):
    return 'The Farmer, fox, chicken, and grain are all across the river!'

def get_name(actors, src, dst):
    actorMap = {'F': 'Farmer', 'f': 'fox', 'c': 'chicken', 'g': 'grain'}
    actorString = ' and '.join([actorMap[a] for a in actors])
    return 'Move ' + actorString + ' from ' + src + ' to ' + dst + '.'


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
CREATE_INITIAL_STATE = lambda : State(INITIAL_DICT)
#</INITIAL_STATE>

#<OPERATORS>
actions = ((['F'],      'L', 'R'),
           (['F', 'f'], 'L', 'R'),
           (['F', 'c'], 'L', 'R'), 
           (['F', 'g'], 'L', 'R'), 
           (['F'],      'R', 'L'),
           (['F', 'f'], 'R', 'L'),
           (['F', 'c'], 'R', 'L'),  
           (['F', 'g'], 'R', 'L'))

OPERATORS = [Operator(get_name(actors, src, dst),
                      lambda state, a=actors, s=src, d=dst : state.can_move(a, s, d),
                      lambda state, a=actors, s=src, d=dst : state.move(a, s, d))
            for (actors, src, dst) in actions] 
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>