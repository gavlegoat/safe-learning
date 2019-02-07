from main import *

from Environment import Environment
from shield import Shield
from DDPG import *

def quadcopter (learning_method, number_of_rollouts, simulation_steps, K=None):
    A = np.matrix([[1,1], [0,1]])
    B = np.matrix([[0],[1]])

    #intial state space
    s_min = np.array([[-0.5],[-0.5]])
    s_max = np.array([[ 0.5],[ 0.5]])

    # LQR quadratic cost per state
    Q = np.matrix("1 0; 0 0")
    R = np.matrix("1.0")

    x_min = np.array([[-1.],[-1.]])
    x_max = np.array([[ 1.],[ 1.]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R)

    args = { 'actor_lr': 0.001,
             'critic_lr': 0.01,
             'actor_structure': [240,200],
             'critic_structure': [280,240,200], 
             'buffer_size': 1000000,
             'gamma': 0.99,
             'max_episode_len': 100,
             'max_episodes': 0,
             'minibatch_size': 64,
             'random_seed': 6553,
             'tau': 0.005,
             'model_path': "ddpg_chkp/quadcopter/240200280240200/"+"model.chkp",
             'enable_test': True, 
             'test_episodes': 10,
             'test_episodes_len': 5000}
    actor = DDPG(env, args=args)
    actor_boundary(env, actor)

    ################# Shield ######################
    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    shield = Shield(env, actor, model_path, force_learning=False, debug=False)
    shield.train_shield(learning_method, number_of_rollouts, simulation_steps)
    shield.test_shield(10, 5000, mode="single")
    shield.test_shield(10, 5000, mode="all")

quadcopter ("random_search", 50, 100) 