import numpy as np
import time

def distance_between_linear_function_and_neural_network(env, actor, K, terminal_err=0.01, rounds=10, steps=500):
	"""sum distance between the output of LF and NN 
		until the state's MSE*dim less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    K (numpy.matrix): coefficient of LF
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	distance = 0
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err

	for i in range(rounds):
		env.reset()
		ep_distance = 0
		for s in range(steps):
			xk, r, terminal = env.observation()
			if r == env.bad_reward:
				sum_steps -= s
				distance -= ep_distance
			if terminal:
				break
			sum_steps += 1
			u1 = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			u2 = K.dot(xk)
			env.step(u1)
			distance += np.linalg.norm(u1-u2)
			ep_distance += np.linalg.norm(u1-u2)

	env.terminal_err = temp_env_ter_err
	if sum_steps == 0:
		return 1
	return float(distance)/sum_steps


def neural_network_performance(env, actor, terminal_err=0.01, rounds=10, steps=500):
	"""Measured by the steps NN took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    actor (DDPG.ActorNetwork): actor
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err
	success_rounds = rounds

	for i in range(rounds):
		env.reset()
		for s in range(steps):
			xk, r, terminal = env.observation()
			if r == env.bad_reward:
				sum_steps -= s
				success_rounds -= 1
			if terminal:
				break
			sum_steps += 1
			u = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
			env.step(u)

	env.terminal_err = temp_env_ter_err
	if success_rounds == 0:
		return steps+1

	return float(sum_steps)/success_rounds

def linear_function_performance(env, K, terminal_err=0.01, rounds=100, steps=500):
	"""Measured by the steps LF took until 
	the sum of state absolute value less than terminal_err
	
	Args:
	    env (DDPG.Enviorment): Enviorment
	    K (numpy.matrix): coefficient of LF
	    terminal_err(float): when terminal
	    rounds(int): rounds
	    steps(int): steps
	"""
	sum_steps = 0
	temp_env_ter_err = env.terminal_err
	env.terminal_err = terminal_err
	for i in range(rounds):
		env.reset()
		for s in range(steps):
			xk, r, terminal = env.observation()
			if terminal:
				break
			sum_steps += 1
			u = K.dot(xk)
			env.step(u)

	env.terminal_err = temp_env_ter_err
	return float(sum_steps)/rounds

def timeit(func):
	"""Record time a function runs with, print it to standard output
	
	Args:
	    func (callable): The function measured
	"""
	def wrapper(*args, **kvargs):
		start = time.time()
		ret = func(*args, **kvargs)
		end = time.time()
		t = end-start
		print func.__name__, 'run time:', t, 's'
		return ret

	return wrapper


def find_boundary(x, x_max, x_min):
    """find if x is between x_max and x_min
    if not, extending x_max and x_min with x
    
    Args:
        x (np.array): state
        x_max (np.array): state max values
        x_min (np.array): state min values
    """
    max_update = (x > x_max)
    min_update = (x < x_min)
    x_max = np.multiply(x,max_update) + np.multiply(x_max, np.logical_not(max_update))
    x_min = np.multiply(x,min_update) + np.multiply(x_min, np.logical_not(min_update))

    return x_max, x_min

