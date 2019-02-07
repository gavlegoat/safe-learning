from DDPG import *
from main import *
import os.path
from Environment import Environment
from shield import Shield

def dcmotor (learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  A = np.matrix([[0.98965,1.4747e-08],
    [7.4506e-09,0]
    ])

  B = np.matrix([[128],
    [0]
    ])

  #intial state space
  s_min = np.array([[-1.0],[-1.0]])
  s_max = np.array([[ 1.0],[ 1.0]])

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-1.5],[-1.5]])
  x_max = np.array([[ 1.5],[ 1.5]])
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
           'enable_test': False, 
           'test_episodes': 100,
           'test_episodes_len': 500}
  actor = DDPG(env, args)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, explore_mag = 0.0004, step_size = 0.0005)
  shield.test_shield(100, 500, mode="single")
  shield.test_shield(100, 500, mode="all")

dcmotor("random_search", 100, 200, 0, [240,200], [280,240,200], "ddpg_chkp/dcmotor/240200280240200/")