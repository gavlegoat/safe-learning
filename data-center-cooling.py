from main import *

from shield import Shield
from Environment import Environment
from DDPG import *

def cooling (learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
    A = np.matrix([
      [1.01,0.01,0],
      [0.01,1.01,0.01],
      [0.0,0.01,1.01]])
    B = np.matrix([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]])

    #intial state space
    s_min = np.array([[  1.6],[ 1.6], [1.6]])
    s_max = np.array([[  3.2],[ 3.2], [3.2]])

    Q = np.eye(3)
    R = np.eye(3)

    x_min = np.array([[-3.2],[-3.2],[-3.2]])
    x_max = np.array([[3.2],[3.2],[3.2]])
    u_min = np.array([[-1.],[-1.],[-1.]])
    u_max = np.array([[ 1.],[ 1.],[ 1.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, bad_reward=-1000)

    args = { 'actor_lr': 0.001,
             'critic_lr': 0.01,
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
             'test_episodes_len': 1000}
    actor = DDPG(env, args)

    #################### Shield #################
    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    names = {0:"cart position, meters", 1:"cart velocity", 2:"pendulum angle, radians", 3:"pendulum angle velocity"}
    shield = Shield(env, actor, model_path)
    shield.train_shield(learning_method, number_of_rollouts, simulation_steps, names=names, explore_mag = 0.02, step_size = 0.0025)
    shield.test_shield(10, 1000, mode="single")
    # shield.test_shield(100, 1000, mode="all")

cooling("random_search", 100, 100, 0, [240, 200], [280, 240, 200], "ddpg_chkp/cooling/240200280240200/")