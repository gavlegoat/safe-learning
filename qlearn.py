import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

class Env():
	def __init__(self):
		l = .22 # rod length is 2l
		m = (2*l)*(.006**2)*(3.14/4)*7856 # rod 6 mm diameter, 44cm length, 7856 kg/m^3
		M = .4
		dt = .02 # 20 ms
		g = 9.8

		self.A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
		self.B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])

		# initial condition for system
		self.x0 = np.matrix(".0; .0; .0; .1")

		self.Q = np.matrix("1 0 0 0; 0 .0001 0 0 ; 0 0 1 0; 0 0 0 .0001")
		self.R = np.matrix(".0005")

		self.x = self.x0
		self.c = 0

	def state(self):
		state = [self.x.item(0,0), self.x.item(1,0), self.x.item(2,0), self.x.item(3,0)]
		return state

	def reset(self):
		self.x = self.x0
		self.c = 0
		return self.state()

	def action_space_n(self):
		return 3

	def failed(self):
		return abs(self.x[0]) > 0.5 or abs(self.x[3]) > 0.5

	def done(self):
		return False

	def step(self, action):
		if (action == 0):
			action = np.matrix("-1")
		elif (action == 1):
			action = np.matrix("0")
		else:
			action = np.matrix("1")
		self.x = self.A.dot(self.x)+self.B.dot(action)
		reward = -np.dot(self.x.T,self.Q.dot(self.x))-np.dot(action.T,self.R.dot(action))
		reward = reward.item(0, 0)
		done = self.done()
		failed = self.failed()
		return self.state(), reward, done, failed

class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, env):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space_n()):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            s = env.reset()
            model.partial_fit([s], [0])
            self.models.append(model)
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all

       	in the environment where pred[i] is the prediction for action i.
            
        """
        if not a:
            return np.array([m.predict([s])[0] for m in self.models])
        else:
            return self.models[a].predict([s])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        self.models[a].partial_fit([s], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
        
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space_n())
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()

        # Only used for SARSA, not Q-Learning
        next_action = None
        
        step = 0

        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, failed = env.step(action)
            #print "step reward = {}".format(reward)

            step = step + 1
    
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            estimator.update(state, action, td_target)
            
            #print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward))
                
            if done or failed:
                break
                
            state = next_state
        if (step >= 100):
        	print "episode steps = {}".format(step)
    
    return stats

env = Env()

estimator = Estimator(env)

stats = q_learning(env, estimator, 10000, epsilon=0.0)

plotting.plot_episode_stats(stats, smoothing_window=25)