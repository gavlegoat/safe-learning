from main import *
from Environment import Environment
from DDPG import *
from shield import Shield
import argparse

def cruise (learning_method, number_of_rollouts, simulation_steps,learning_eposides, critic_structure, actor_structure, train_dir,\
            nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100):
  A = np.matrix([[0.99501]])

  B = np.matrix([[0.0078125]])

  #intial state space
  s_min = np.array([[-1.0]])
  s_max = np.array([[ 1.0]])

  Q = np.matrix("1")
  R = np.matrix(".0005")

  x_min = np.array([[-1.5]])
  x_max = np.array([[ 1.5]])
  u_min = np.array([[-10.]])
  u_max = np.array([[ 10.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

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
           'enable_test': nn_test, 
           'test_episodes': test_episodes,
           'test_episodes_len': 5000}

  actor = DDPG(env, args)
  
  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=retrain_shield, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=0, explore_mag = 1.0, step_size = 1.0)
  if shield_test:
    shield.test_shield(test_episodes, 5000, mode="single")

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

  cruise("random_search", 200, 100, 0, [240,200], [280,240,200], "ddpg_chkp/cruise/240200280240200/", nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes)