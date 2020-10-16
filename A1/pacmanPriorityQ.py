''' pacmanPriorityQ.py

This Python 3.8 module provides a priority queue data structure
that is well-suited for AI search algorithms such as UCS and A*.

'''

VERBOSE = False

class AI_Priority_Queue:
  def __init__(self):
    if VERBOSE: print("AI_Priorty_Queue: instantiated")
    self.q = [] # Actual data goes in a list.

  def __contains__(self, elt):
    '''If there is a (state, priority) pair on the list
    where state==elt, then return True.'''
    for pair in self.q:
      if pair[0]==elt: return True
    return False

  def delete_min(self):
    ''' Standard priority-queue dequeuing method.'''
    if self.q==[]: return [] # Simpler than raising an exception.
    temp_min_pair = self.q[0]
    temp_min_value = temp_min_pair[1]
    temp_min_position = 0
    for j in range(1, len(self.q)):
      if self.q[j][1] < temp_min_value:
        temp_min_pair = self.q[j]
        temp_min_value = temp_min_pair[1]  
        temp_min_position = j
    del self.q[temp_min_position]
    if VERBOSE: print("AI_Priorty_Queue.delete_min returns"+str(temp_min_pair))
    return temp_min_pair

  def insert(self, state, priority):
    '''We do not keep the list sorted, in this implementation.'''
    if VERBOSE: print("AI_Priorty_Queue.insert called with a state having priority "+str(priority))
    if self[state] != -1:
      print("Error: You're trying to insert an element into an AI_Priority_Queue instance,")
      print(" but there is already such an element in the queue.")
      return
    self.q.append((state, priority))

  def __len__(self):
    '''We define length of the priority queue to be the length of its list.'''
    return len(self.q)

  def __getitem__(self, state):
    '''This method enables Pythons right-bracket syntax.
    Here, something like  priority_val = my_queue[state]
    becomes possible. Note that the syntax is actually used
    in the insert method above:  self[state] != -1  '''
    for (S,P) in self.q:
      if S==state: return P
    return -1  # This value means not found.

  def __delitem__(self, state):
    '''This method enables Python's del operator to delete
    items from the queue.'''
    for count, (S,P) in enumerate(self.q):
      if S==state:
        del self.q[count]
        return

  def __str__(self):
    "Code to create a string representation of the PQ."
    txt = "AI_Priority_Queue: ["
    for (s,p) in self.q: txt += '('+str(s)+','+str(p)+') '
    txt += ']'
    return txt

if __name__=='__main__':
  # A simple test.
  VERBOSE = True
  instance = AI_Priority_Queue()
  instance.insert("State-0",4)
  print(instance)
  instance.delete_min()

