from main import *
from Environment import Environment
from DDPG import *
from shield import Shield

def magneticpointer (learning_method, number_of_rollouts, simulation_steps,learning_eposides, critic_structure, actor_structure, train_dir, K=None):
  A = np.matrix([[2.6629,-1.1644,0.66598],
    [2, 0, 0],
    [0, 0.5, 0]
    ])

  B = np.matrix([[0.25],
    [0],
    [0]
    ])

  #intial state space
  s_min = np.array([[-1.0],[-1.0], [-1.0]])
  s_max = np.array([[ 1.0],[ 1.0], [ 1.0]])


  Q = np.matrix("1 0 0 ; 0 1 0; 0 0 1")
  R = np.matrix("1")

  x_min = np.array([[-3.5],[-3.5],[-3.5]])
  x_max = np.array([[ 3.5],[ 3.5],[ 3.5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

  args = { 'actor_lr': 0.0001,
           'critic_lr': 0.001,
           'actor_structure': actor_structure,
           'critic_structure': critic_structure, 
           'buffer_size': 1000000,
           'gamma': 0.99,
           'max_episode_len': 20,
           'max_episodes': learning_eposides,
           'minibatch_size': 64,
           'random_seed': 6553,
           'tau': 0.005,
           'model_path': train_dir+"model.chkp",
           'enable_test': True, 
           'test_episodes': 1,
           'test_episodes_len': 500}

  actor = DDPG(env, args)
  
  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path, force_learning=False, debug=False)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=0, explore_mag = 1.0, step_size = 1.0)
  shield.test_shield(1, 500, mode="single")
  # shield.test_shield(1, 500, mode="all")


magneticpointer("random_search", 100, 50, 0, [240,200], [280,240,200], "ddpg_chkp/magnetic_pointer/240200280240200/")