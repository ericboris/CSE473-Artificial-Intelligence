class MDP:
	def __init__(self, transitions, gamma):
		self.transitions = transitions
		self.vt = self.newVT()
		self.qt = self.newQT()
		self.gamma = gamma
	
	def newQT(self):
		t = {'A': {'L': 0, 'R': 0}, 
			'B': {'L': 0, 'R': 0},
			'C': {'L': 0, 'R': 0},
			'D': {'L': 0, 'R': 0},
			'E': {'L': 0, 'R': 0},
			'Z': {'L': 0, 'R': 0}}
		return t

	def newVT(self):
		t = {'A': 0,
			'B': 0,
			'C': 0,
			'D': 0,
			'E': 0,
			'Z': 0}
		return t
	
	def vUpdate(self, n):
		states = ['A', 'B', 'C', 'D', 'E']
		actions = ['L', 'R']
		for i in range(n):
			for s in states:		
				ma = None
				mv = float('-inf')
				for a in actions:
					qval = self.q(s, a)
					if qval > mv:
						ma = a
						mv = qval
				self.qt[s][ma] = mv
				self.vt[s] = mv
			print('k : ', str(i + 1))
			print('qt : ', self.qt)
			print('vt : ', self.vt)
			print('\n')
			
	def q(self, s, a):
		sp, r = self.transitions[s][a]
		return r + self.gamma * self.vt[sp]



TRANSITIONS = {'A': {'L': ('Z', 10), 'R': ('B', -1)}, 
	'B': {'L': ('A', -1), 'R': ('C', -1)},
	'C': {'L': ('B', -1), 'R': ('D', -1)},
	'D': {'L': ('C', -1), 'R': ('E', -1)},
	'E': {'L': ('D', -1), 'R': ('Z', 1)},
	'Z': {'L': ('Z', -1), 'R': ('Z', -1)}}

GAMMA = 0.1

mdp = MDP(TRANSITIONS, GAMMA)

print('k : 0')
print('qt : ', mdp.qt)
print('vt : ', mdp.vt)
print('\n')
mdp.vUpdate(3)
