'''VI.py

Value Iteration for Markov Decision Processes.
'''


# Edit the returned name to ensure you get credit for the assignment.
def student_name():
	return 'Boris, Eric'


Vkplus1 = {}
Q_Values_Dict = {}

def one_step_of_VI(S, A, T, R, gamma, Vk):
	'''S is list of all the states defined for this MDP.
	A is a list of all the possible actions.
	T is a function representing the MDP's transition model.
	R is a function representing the MDP's reward function.
	gamma is the discount factor.
	The current value of each state s is accessible as Vk[s].
	'''

	'''Your code should fill the dictionaries Vkplus1 and Q_Values_dict
	with a new value for each state, and each q-state, and assign them
	to the state's and q-state's entries in the dictionaries, as in
		Vkplus1[s] = new_value
		Q_Values_Dict[(s, a)] = new_q_value

	Also determine delta_max, which we define to be the maximum
	amount that the absolute value of any state's value is changed
	during this iteration.
	'''
	global Vkplus1
	delta_max = 0	
	
	Q_Values_Dict = return_Q_values(S, A)
	s_primes = return_s_primes(S, A)
	
	# Calculate Q(s, a) for every state-action pair.	
	for s in S:
		for a in A:
			Q_Values_Dict[(s, a)] = sum([T(s, a, sp) * (R(s, a, sp) + gamma * Vk[sp]) for sp in s_primes])
				
	
	# Calculate V(s) for every state s by finding the max Q(s, a).
	# Also calculate delta_max.
	for s in S:
		new_value = max([Q_Values_Dict[(s, a)] for a in A])
		dv = abs(Vk[s] -  new_value)
		delta_max = max(dv, delta_max)
		Vkplus1[s] = new_value

	return (Vkplus1, delta_max)

def return_s_primes(S, A):
	''' Return a dictionary mapping s to a list of all s' reachable from s.
	The values passed for S and A are ignored except in the case that s_primes 
	hasn't been initialized in which case they are used to initialize s_primes.
	'''
	s_primes = {}
	for s in S:
		primes = [s]
		for a in A:
			a_str = a.split(' ')
			try:
				src, dst = a_str[3], a_str[5]
			except:	
				continue
			if s.can_move(src, dst):
				sp = s.move(src, dst)
				primes.append(sp)
		s_primes[s] = primes

	return s_primes

def return_Q_values(S, A):
	'''Return the dictionary whose keys are (state, action) tuples,
	and whose values are floats representing the Q values from the
	most recent call to one_step_of_VI. This is the normal case, and
	the values of S and A passed in here can be ignored.
	However, if no such call has been made yet, use S and A to
	create the answer dictionary, and use 0.0 for all the values.
	'''
	global Q_Values_Dict

	# Initialize the dictionary if it hasn't been initialized yet.
	if len(Q_Values_Dict) == 0:
		for s in S:
			for a in A:
				Q_Values_Dict[(s, a)] = 0.0

	return Q_Values_Dict


Policy = {}


def extract_policy(S, A):
	'''Return a dictionary mapping states to actions. Obtain the policy
	using the q-values most recently computed.  If none have yet been
	computed, call return_Q_values to initialize q-values, and then
	extract a policy.  Ties between actions having the same (s, a) value
	can be broken arbitrarily.
	'''
	global Policy, Q_Values_Dict
	Policy = {}

	actions = A[:]

	for s in S:
		actions.sort(key=lambda a : Q_Values_Dict[(s, a)], reverse=True)
		Policy[s] = actions[0]

	return Policy


def apply_policy(s):
	'''Return the action that your current best policy implies for state s.'''
	global Policy
	return Policy[s]
