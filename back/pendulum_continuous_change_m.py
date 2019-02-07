# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-23 17:04:25
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-11-14 15:14:40
# -------------------------------
from main import *
import numpy as np
from DDPG import *
import sys
from shield import Shield
from Environment import Environment

def pendulum(learning_eposides, critic_structure, actor_structure, train_dir, learning_method, number_of_rollouts, simulation_steps):
  
  m = 1.8
  l = 1.
  g = 10.

  #Dynamics that are continuous
  A = np.matrix([
    [ 0., 1.],
    [g/l, 0.]
    ])
  B = np.matrix([
    [          0.],
    [1./(m*l**2.)]
    ])


  #intial state space
  s_min = np.array([[-0.35],[-0.35]])
  s_max = np.array([[ 0.35],[ 0.35]])

  #reward function
  Q = np.matrix([[1., 0.],[0., 1.]])
  R = np.matrix([[.005]])

  #safety constraint
  x_min = np.array([[-0.5],[-0.5]])
  x_max = np.array([[ 0.5],[ 0.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 1,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': False, 
           'test_episodes': 100,
           'test_episodes_len': 1000}

  actor = DDPG(env, args)
  
  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=False, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=1e-2, explore_mag = 1.0, step_size = 1.0)
  shield.test_shield(1000, 5000, mode="single")
  shield.test_shield(1000, 5000, mode="all")

  ################# Metrics ######################
  # actor_boundary(env, actor, epsoides=500, steps=200)
  # shield.shield_boundary(2000, 50)
  # terminal_err = 0.1
  # sample_steps = 100
  # sample_ep = 1000
  # print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\n---".format(terminal_err, sample_ep, sample_steps)
  # dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
  # print "dist_nn_lf: ", dist_nn_lf
  # nn_perf = metrics.neural_network_performance(env, actor, terminal_err, sample_ep, sample_steps)
  # print "nn_perf", nn_perf
  # shield_perf = metrics.linear_function_performance(env, shield.K, terminal_err, sample_ep, sample_steps)
  # print "shield_perf", shield_perf

  actor.sess.close()

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  pendulum(0, [1200,900], [1000,900,800], "ddpg_chkp/perfect_model/pendulum/change_m/", "random_search", 100, 200) 
