# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-23 22:14:33
# Last Modified by:   Zikang Xiong
# Last Modified time: 2019-01-30 00:42:20
# -------------------------------
from main import *
import sys
from DDPG import *

from Environment import Environment
from shield import Shield

def cartpole(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  A = np.matrix([
  [0, 1,     0, 0],
  [0, 0, 0.716, 0],
  [0, 0,     0, 1],
  [0, 0, 15.76, 0]
  ])
  B = np.matrix([
  [0],
  [0.9755],
  [0],
  [1.46]
  ])

   #intial state space
  s_min = np.array([[ -0.05],[ -0.1], [-0.05], [ -0.05]])
  s_max = np.array([[  0.05],[  0.1], [ 0.05], [  0.05]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-0.3],[-0.5],[-0.3],[-0.5]])
  x_max = np.array([[ .3],[ .5],[.3],[.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])
  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)

  args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 100,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': True, 
         'test_episodes': 1,
         'test_episodes_len': 5000}
  actor =  DDPG(env, args=args)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=0, explore_mag = 1.0, step_size = 1.0)
  # shield.test_shield(1, 5000, mode="single")
  #shield.test_shield(1000, 5000, mode="all")
  #shield.test_shield(x0=np.array([[ 0.04472518],[ 0.06732232],[-0.04092728],[-0.00663184]]), mode="single", loss_compensation=0)

#  ################# Metrics ######################
#  actor_boundary(env, actor, 1000, 100)
#  shield.shield_boundary(1000, 100)
#  terminal_err = 0.1
#  sample_steps = 100
#  sample_ep = 1000
#  print "---\nterminal error: {}\nsample_ep: {}\nsample_steps: {}\n---".format(terminal_err, sample_ep, sample_steps)
#  dist_nn_lf = metrics.distance_between_linear_function_and_neural_network(env, actor, shield.K, terminal_err, sample_ep, sample_steps)
#  print "dist_nn_lf: ", dist_nn_lf
#  nn_perf = metrics.neural_network_performance(env, actor, terminal_err, sample_ep, sample_steps)
#  print "nn_perf", nn_perf
#  shield_perf = metrics.linear_function_performance(env, shield.K, terminal_err, sample_ep, sample_steps)
#  print "shield_perf", shield_perf
#
  actor.sess.close()

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  cartpole("random_search", 100, 200, 0, [300, 200], [300, 250, 200], "ddpg_chkp/cartpole/continuous/300200300250200_auto/")
