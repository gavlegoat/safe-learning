# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-28 16:38:03
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-11-09 02:52:02
# -------------------------------
import numpy as np
from DDPG import *
from main import *
import os.path
import sys

from shield import Shield
from Environment import Environment
import metrics


def pendulum(learning_eposides, actor_structure, critic_structure, train_dir, learning_method, number_of_rollouts, simulation_steps):
  
  ############## Train NN Controller ###############
  # State transform matrix
  A = np.matrix([[1.9027, -1],
                      [1, 0]
                      ])

  B = np.matrix([[1],
                      [0]
                      ])

  # initial action space
  u_min = np.array([[-1.]])
  u_max = np.array([[1.]])

  # intial state space
  s_min = np.array([[-0.5],[-0.5]])
  s_max = np.array([[ 0.5],[0.5]])
  x_min = np.array([[-0.6], [-0.6]])
  x_max = np.array([[0.6], [0.6]])

  # coefficient of reward function
  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix(".0005")

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 500,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': True, 
           'test_episodes': 1000,
           'test_episodes_len': 5000}
  actor = DDPG(env, args=args)
  #actor_boundary(env, actor)

  ################# Shield ######################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=False, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps)
  shield.test_shield(1000, 5000, mode="single")
  shield.test_shield(1000, 5000, mode="all")
  # shield.shield_boundary(2000, 50)

  ################# Metrics ######################
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
  pendulum(0, [1200,900], [1000,900,800], "ddpg_chkp/pendulum/discrete/", "random_search", 100, 50) 