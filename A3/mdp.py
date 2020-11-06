from collections import defaultdict

class QLearn:
	def __init__(self, state, alpha, gamma, transitions):
		self.state = state
		self.alpha = alpha
		self.gamma = gamma
		self.transitions = transitions
		self.qTable = self.newTable() 
		self.vTable = self.newTable()

	def newTable(self):
		qTable = {'A': {'E': 0, 'S': 0},
			'B': {'E': 0, 'S': 0, 'W': 0},
			'C': {'S': 0, 'W': 0},
			'D': {'E': 0, 'S': 0, 'N': 0}, 
			'K': {'E': 0, 'S': 0, 'W': 0, 'N': 0},
			'F': {'S': 0, 'W': 0, 'N': 0}, 
			'G': {'E': 0, 'N': 0, 'X': 0}, 
			'H': {'E': 0, 'W': 0, 'N': 0}, 
			'I': {'W': 0, 'N': 0}}
		return qTable

	def nextAction(self, s):
		possibleActions = ['E', 'S', 'W', 'N', 'X']
		maxQ = float('-inf')
		a = None
		print('next action')
		for pa in possibleActions:
			if pa in self.qTable[s] and self.qTable[s][pa] > maxQ:
				maxQ = self.qTable[s][pa]
				a = pa
				print('state ' + s)
				print('action ' + a)
		return a

	def update(self, s, a, isSimple):
		# Make sure that the action is legal from the current state.
		if a not in self.qTable[self.state]:
			return

		# If it is then move to the new state s' and update the qTable of q values.
		self.state = s		

		# Increment the action taken from state
		self.vTable[s][a] += 1

		# Calculate the sample.
		r, sPrime = self.transitions[s][a]
		if isSimple:
			maxQ = max([v for v in self.qTable[sPrime].values()])
		else:
			f = lambda u, n : u + (1 / (1 + n))
			maxQ = max([f(self.qTable[sPrime][ap], self.vTable[sPrime][ap]) for ap in self.qTable[sPrime].keys()])
		sample = r + self.gamma * maxQ
		
		# Calculate the new Q value.
		currQ = self.qTable[s][a]
		newQ = (1 - self.alpha) * currQ + self.alpha * sample
	
		# Update the qTable with the new Q value.	
		self.qTable[s][a] = newQ	

def display(s, a, sPrime, qOld, qNew):
	''' Print a string detailing state and value changes.'''
	txt = 's a s`:\n'
	txt += s + ' ' + a + ' ' + sPrime + '\n'
	txt += 'Q update:\n'
	txt += str(qOld) + ' -> ' + str(qNew) + '\n'
	print(txt)

def process(actions):
	''' Process the actions in actions, updating a q qTable along the way.'''
	q = QLearn(START_STATE, ALPHA, GAMMA, TRANSITIONS)
	s = START_STATE
	print('-- Begin --\n')	
	for a in actions:	
		_, sPrime = TRANSITIONS[s][a]
		qOld = q.qTable[s][a]
		q.update(s, a, True)
		qNew = q.qTable[s][a]
		display(s, a, sPrime, qOld, qNew)
		s = sPrime
	print('\n')	
    
def explore(steps):
	''' Process the actions in actions, updating a q qTable along the way.'''
	q = QLearn(START_STATE, ALPHA, GAMMA, TRANSITIONS)
	s = START_STATE
	print('-- Begin Explore --\n')	
	for i in range(steps):
		a = q.nextAction(s)
		_, sPrime = TRANSITIONS[s][a]
		qOld = q.qTable[s][a]
		q.update(s, a, False)
		qNew = q.qTable[s][a]
		display(s, a, sPrime, qOld, qNew)
		s = sPrime
	print('\n')
        
    
TRANSITIONS = {'A': {'E': (-1, 'B'), 'S': (1, 'D')},
    'B': {'E': (-1, 'C'), 'S': (-1, 'K'), 'W': (-1, 'A')},
	'C': {'S': (-1, 'F'), 'W': (-1, 'B')},
	'D': {'E': (-1, 'K'), 'S': (1, 'G'), 'N': (-1, 'A')}, 
	'K': {'E': (-10, 'F'), 'S': (-10, 'H'), 'W': (-1, 'D'),'N': (-1, 'B')},
	'F': {'S': (-1, 'I'), 'W': (-1, 'K'), 'N': (-1, 'C')}, 
	'G': {'E': (-10, 'H'), 'N': (-1, 'D'), 'X': (5, 'Z')}, 
	'H': {'E': (-10, 'I'), 'W': (5, 'G'), 'N': (-1, 'K')}, 
	'I': {'W': (5, 'H'), 'N': (-1, 'F')}}

ALPHA = 0.5
GAMMA = 1

A = ['E', 'S', 'W', 'N', 'E', 'S', 'W', 'N']
B = ['E', 'E', 'W', 'S', 'E', 'N', 'W', 'W']
X = ['E', 'E', 'S', 'S', 'W', 'E', 'W', 'W', 'N', 'E', 'W', 'E', 'E', 'S','W', 'E']
SECTIONS = [A, B, X]

EXPLORE_SECTIONS = [8, 16]

START_STATE = 'A'

for actions in SECTIONS:
	process(actions)	

for e in EXPLORE_SECTIONS:
    explore(e)
