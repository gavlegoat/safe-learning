import numpy as np
from DDPG import *
from main import *
import os.path
import argparse

from Environment import Environment
from shield import Shield


def carplatoon(learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir, \
              nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100):
  A = np.matrix([
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,1,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,1,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,1,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,1, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,1, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,1, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,1],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0]
  ])
  B = np.matrix([
  [1,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [1,  -1,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   1,  -1,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   1,  -1,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   1,  -1,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   1,  -1,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   1,  -1,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   1,  -1],
  ])

  #intial state space
  s_min = np.array([[ 19.9],[ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1]])
  s_max = np.array([[ 20.1],[ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1]])

  x_min = np.array([[18],[0.1],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1]])
  x_max = np.array([[22],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5],[ 1]])
  u_min = np.array([[-10.], [-10.], [-10.], [-10.], [-10.], [-10.], [-10.], [-10.]])
  u_max = np.array([[ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.], [ 10.]])

  target = np.array([[20],[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

  s_min -= target
  s_max -= target
  x_min -= target
  x_max -= target

  Q = np.zeros((15, 15), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((8,8), float)
  np.fill_diagonal(R, 1)

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True, bad_reward=-1000)
  args = { 'actor_lr': 0.000001,
           'critic_lr': 0.00001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.999,
           'max_episode_len': 400,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 122,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': nn_test, 
           'test_episodes': test_episodes,
           'test_episodes_len': 1200}
  actor = DDPG(env, args)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  def rewardf(x, Q, u, R):
    return env.reward(x, u)

  names = {0:"x0", 1:"x1", 2:"x2", 3:"x3", 4:"x4", 5:"x5", 6:"x6", 7:"x7", 8:"x8", 9:"x9", 10:"x10", 11:"x11", 12:"x12", 13:"x13", 14:"x14"}
  shield = Shield(env, actor, model_path, force_learning=retrain_shield)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, rewardf=rewardf, names=names, explore_mag = 0.1, step_size = 0.1)
  if shield_test:
    shield.test_shield(test_episodes, 1200)

  actor.sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Running Options')
  parser.add_argument('--nn_test', action="store_true", dest="nn_test")
  parser.add_argument('--retrain_shield', action="store_true", dest="retrain_shield")
  parser.add_argument('--shield_test', action="store_true", dest="shield_test")
  parser.add_argument('--test_episodes', action="store", dest="test_episodes", type=int)
  parser_res = parser.parse_args()
  nn_test = parser_res.nn_test
  retrain_shield = parser_res.retrain_shield
  shield_test = parser_res.shield_test
  test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100

  carplatoon("random_search", 500, 2000, 0, [400, 300, 200], [500, 400, 300, 200], "ddpg_chkp/car-platoon/continuous/8/400300200500400300200/", 
              nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes) 