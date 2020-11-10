'''Q_Learn.py

Implement Q-Learning in this file by completing the implementations
of the functions whose stubs are present.
Add or change code wherever you see #*** ADD OR CHANGE CODE HERE ***

This is part of the UW Intro to AI Starter Code for Reinforcement Learning.

'''
import random
import math


# Edit the returned name to ensure you get credit for the assignment.
def student_name():
    # *** ADD OR CHANGE CODE HERE ***
    return "Your Lastname, Firstname"  # For an autograder.


STATES = None
ACTIONS = None
UQV_callback = None
Q_VALUES = None
is_valid_goal_state = None
Terminal_state = None
INITIAL_STATE = None
TIME = 1  # Number of Q-Learning iterations so far. You should not change this variable for autograding purpose


def setup(states, actions, q_vals_dict, update_q_value_callback, goal_test, terminal, use_exp_fn=False):
    '''This method is called by the GUI the first time a Q_Learning
    menu item is selected. It may be called again after the user has
    restarted from the File menu.
    Q_VALUES starts out with all Q-values at 0.0 and a separate key
    for each (s, a) pair.'''
    global STATES, ACTIONS, UQV_callback, Q_VALUES, is_valid_goal_state
    global USE_EXPLORATION_FUNCTION, Terminal_state
    STATES = states
    ACTIONS = actions
    Q_VALUES = q_vals_dict
    UQV_callback = update_q_value_callback
    is_valid_goal_state = goal_test
    Terminal_state = terminal


PREVIOUS_STATE = None
LAST_ACTION = None


def set_starting_state(s):
    '''This is called by the GUI when a new episode starts.
    Do not change this function.'''
    global INITIAL_STATE, PREVIOUS_STATE, TIME
    print("In Q_Learn, setting the starting state to " + str(s))
    INITIAL_STATE = s
    PREVIOUS_STATE = s
    TIME = 1


ALPHA = 0.5
CUSTOM_ALPHA = False
EPSILON = 0.5
CUSTOM_EPSILON = False
GAMMA = 0.9


def set_learning_parameters(alpha, epsilon, gamma):
    ''' Called by the system. Do not change this function.'''
    global ALPHA, EPSILON, CUSTOM_ALPHA, CUSTOM_EPSILON, GAMMA
    ALPHA = alpha
    EPSILON = epsilon
    GAMMA = gamma
    if alpha < 0:
        CUSTOM_ALPHA = True
    else:
        CUSTOM_ALPHA = False
    if epsilon < 0:
        CUSTOM_EPSILON = True
    else:
        CUSTOM_EPSILON = False


def update_vis_Q_value(previous_state, previous_action, new_value):
    '''Whenever your code changes a value in Q_VALUES, it should
    also call this method, so the changes can be reflected in the
    display.
    Do not change this function.'''
    UQV_callback(previous_state, previous_action, new_value)


def handle_transition(action, new_state, r):
    '''When the user drives the agent, the system will call this function,
    so that you can handle the learning that should take place on this
    transition.'''
    global PREVIOUS_STATE, LAST_ACTION, Q_VALUES
    # Compute your update here.
    # if CUSTOM_ALPHA is True, use alpha value (1 / TIME)
    # Otherwise go with the fixed value.

    # *** ADD OR CHANGE CODE HERE ***
    new_qval = -99  # A bogus value for now.

    # Save it in the dictionary of Q_VALUES:
    Q_VALUES[(PREVIOUS_STATE, action)] = new_qval

    # Then let the Engine and GUI know about the new Q-value.
    update_vis_Q_value(PREVIOUS_STATE, action, new_qval)

    # Comment out once finish
    print("Transition to state: " + str(new_state) + \
          "\n with reward " + str(r) + " is currently not handled by your program.")

    LAST_ACTION = action
    PREVIOUS_STATE = new_state
    return  # Nothing needs to be returned.


def rand_action(optimal_actions, epsilon, valid_actions):
    '''
    optimal_actions is a list of actions with the hightest q-value.
     If there's a unique optimal action, this will be a one element list
    epsilon is the probability of exploration
    valid_actions is a list actions that the current state can take
    '''
    global TIME
    random.seed(TIME)
    if random.random() >= epsilon:
        optimal_actions = optimal_actions.copy()
        optimal_actions.sort()
        return random.choice(optimal_actions)
    else:
        valid_actions = valid_actions.copy()
        valid_actions.sort()
        return random.choice(valid_actions)


def choose_next_action(s, r, terminated=False):
    '''When the GUI or engine calls this, the agent is now in state s,
    and it receives reward r.
    If terminated==True, or the state is the terminal state,
     it's the end of the episode, and this method
     can just return None after you have handled the transition.

    If reached goal state, return the action 'Exit'

    Use this information to update the q-value for the previous state
    and action pair.

    Then the agent needs to choose its action and return that.

    '''
    global INITIAL_STATE, LAST_ACTION, TIME
    # Unless s is the initial state, compute a new q-value for the
    # previous state and action.
    if not (s == INITIAL_STATE):
        # *** ADD OR CHANGE CODE HERE ***
        raise NotImplementedError("To be implemented")

    # Check for terminated and goal state here
    # *** ADD OR CHANGE CODE HERE ***
    raise NotImplementedError("To be implemented")

    # Now select an action according to your Q-Learning criteria, such
    # as expected discounted future reward vs exploration.
    # If CUSTOM_EPSILON is True,
    # then use epsilon of sqrt(1 / TIME)
    # Otherwise, use the constant epsilon here.
    # In order to access q-values, simply get them from the dictionary, e.g.,
    # some_qval = Q_VALUES[(some_state, some_action)]

    # *** USE rand_action HERE! ***
    # The grading relies on the fixed randomness of rand_action.
    # If you fail to use it, you will not get points for this problem.
    # Make sure you don't remove from ACTIONS here, remove from a copy of it instead

    # *** ADD OR CHANGE CODE HERE ***
    raise NotImplementedError("To be implemented")
    some_action = None


    # Update global variables
    LAST_ACTION = some_action  # remember this for next time
    TIME += 1  # You should not change this variable for autograding purpose
    return some_action


Policy = {}



def extract_policy(S, A):
    '''Return a dictionary mapping states to actions. Obtain the policy
    using the q-values most recently computed.
    Ties between actions having the same (s, a) value can be broken arbitrarily.
    Reminder: goal states should map to the Exit action, and no other states
    should map to the Exit action.
    '''
    global Policy
    Policy = {}
    # *** ADD OR CHANGE CODE HERE ***
    return Policy
