from collections import defaultdict

class MDP:
	def __init__(self, state, alpha, gamma, transitions):
		self.state = state
		self.alpha = alpha
		self.gamma = gamma
		self.transitions = transitions
		self.q = self.newTable() 
		self.n = self.newTable()

	def newTable(self):
		''' Return a new state-action table with values initalized to 0. '''
		t = {'A': {'E': 0, 'S': 0},
			'B': {'E': 0, 'S': 0, 'W': 0},
			'C': {'S': 0, 'W': 0},
			'D': {'E': 0, 'S': 0, 'N': 0}, 
			'K': {'E': 0, 'S': 0, 'W': 0, 'N': 0},
			'F': {'S': 0, 'W': 0, 'N': 0}, 
			'G': {'E': 0, 'N': 0, 'X': 0}, 
			'H': {'E': 0, 'W': 0, 'N': 0}, 
			'I': {'W': 0, 'N': 0}}
		return t 

	def nextAction(self, s):
		''' Return the highest valued action to take from the current step. '''
		possibleActions = ['E', 'S', 'W', 'N', 'X']
		maxQ = float('-inf')
		a = None
		for pa in possibleActions:
			if pa in self.q[s] and self.q[s][pa] > maxQ:
				maxQ = self.q[s][pa]
				a = pa
		return a

	def regularUpdate(self, s, a):
		''' Find the utility of performing action a from state s. '''
		r, sp = self.transitions[s][a]
		maxUtility = max([u for u in self.q[sp].values()])
		sample = r + self.gamma * maxUtility
		newUtility = (1 - self.alpha) * self.q[s][a] + self.alpha * sample
		return newUtility
    
	def modifiedUpdate(self, s, a):
		''' Find the utility of performing action a from state s with an explore function. '''
		if s == 'Z':
			return
		r, sp = self.transitions[s][a]
		f = lambda u, n: u + (1 / (n + 1))
		maxF = float('-inf')
		for ap in self.transitions[sp].keys():
			u = self.q[sp][ap]
			n = self.n[sp][ap]
			if f(u, n) > maxF:
				maxF = f(u, n)
		sample = r + self.gamma * maxF
		newUtility = (1 - self.alpha) * self.q[s][a] + self.alpha * sample
		return newUtility

	def updateQ(self, s, a, utility):
		''' Assign the given utility value to state s and action a. '''
		self.q[s][a] = utility

	def changeState(self, s, a):
		''' Update the model to reflect a change in state from s via action a to state sp. '''
		_, sp = self.transitions[s][a]
		self.state = sp

	def incrementVisit(self, s, a):
		''' Increment the number of visits to state s. '''
		self.n[s][a] += 1

	def currentState(self):
		''' Return the current state of the mdp. '''
		return self.state

	def currentQ(self, s, a):
		''' Return the current utility value at s, a. '''
		return self.q[s][a]


def display(s, a, sp, mdpOld, mdpNew):
	''' Print a string detailing state and value changes.'''
	txt = 's a s`:\n'
	txt += s + ' ' + a + ' ' + sp + '\n'
	txt += 'Q update:\n'
	txt += str(mdpOld) + ' -> ' + str(mdpNew) + '\n'
	print(txt)

def process(actions):
	''' Perform the given actions with a regular q update. '''
	mdp = MDP(START_STATE, ALPHA, GAMMA, TRANSITIONS)
	s = START_STATE
	print('-- Begin --\n')	
	for a in actions:	
		oldQ = mdp.currentQ(s, a)
		newQ = mdp.regularUpdate(s, a)
		mdp.updateQ(s, a, newQ)
		mdp.changeState(s, a)
		sp = mdp.currentState()
		display(s, a, sp, oldQ, newQ)
		s = sp
	print(mdp.q)
	print('\n')	

def explore(k):
	''' Perform k iterations of the modified q update. '''
	mdp = MDP(START_STATE, ALPHA, GAMMA, TRANSITIONS)
	s = START_STATE
	print('-- Begin Explore --\n')
	for i in range(k):
		a = mdp.nextAction(s)
		oldQ = mdp.currentQ(s, a)
		newQ = mdp.modifiedUpdate(s, a)
		mdp.updateQ(s, a, newQ)
		mdp.incrementVisit(s, a)
		mdp.changeState(s, a)
		sp = mdp.currentState()
		display(s, a, sp, oldQ, newQ)
		s = sp
	print(mdp.q)
	print(mdp.n)
	print('\n')

START_STATE = 'A'
ALPHA = 0.5
GAMMA = 1

TRANSITIONS = {'A': {'E': (-1, 'B'), 'S': (1, 'D')},
    'B': {'E': (-1, 'C'), 'S': (-1, 'K'), 'W': (-1, 'A')},
	'C': {'S': (-1, 'F'), 'W': (-1, 'B')},
	'D': {'E': (-1, 'K'), 'S': (1, 'G'), 'N': (-1, 'A')}, 
	'K': {'E': (-10, 'F'), 'S': (-10, 'H'), 'W': (-1, 'D'),'N': (-1, 'B')},
	'F': {'S': (-1, 'I'), 'W': (-1, 'K'), 'N': (-1, 'C')}, 
	'G': {'E': (-10, 'H'), 'N': (-1, 'D'), 'X': (5, 'Z')}, 
	'H': {'E': (-10, 'I'), 'W': (5, 'G'), 'N': (-1, 'K')}, 
	'I': {'W': (5, 'H'), 'N': (-1, 'F')}}

A = ['E', 'S', 'W', 'N', 'E', 'S', 'W', 'N']
B = ['E', 'E', 'W', 'S', 'E', 'N', 'W', 'W']
X = ['E', 'E', 'S', 'S', 'W', 'E', 'W', 'W', 'N', 'E', 'W', 'E', 'E', 'S','W', 'E']
SECTIONS = [A, B, X]

C_AND_X = [8, 16]

for actions in SECTIONS:
	process(actions)	

for k in C_AND_X:
    explore(k)
