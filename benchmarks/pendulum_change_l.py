import sys
sys.path.append(".")

from main import *
import numpy as np
from DDPG import *

from shield import Shield
from Environment import Environment
import argparse

def pendulum(learning_episodes, critic_structure, actor_structure, train_dir,
        learning_method, number_of_rollouts, simulation_steps, nn_test=False,
        retrain_shield=False, shield_test=False, test_episodes=100,
        retrain_nn=False, safe_training=False, shields=1, episode_len=500,
        penalty_ratio=0.1):

    m = 1.
    l = 1.2
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

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R,
            continuous=True)

    x_mid = (x_min + x_max) / 2.0
    def safety_reward(x, Q, u, R):
        return -np.matrix([[np.sum(np.abs(x - x_mid))]])

    if retrain_nn:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure, 
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,
                 'max_episodes': learning_episodes,
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"retrained_model.chkp",
                 'enable_test': nn_test, 
                 'test_episodes': test_episodes,
                 'test_episodes_len': 3000}
    else:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure, 
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,
                 'max_episodes': learning_eposides,
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"model.chkp",
                 'enable_test': nn_test, 
                 'test_episodes': test_episodes,
                 'test_episodes_len': 3000}

    Ks = [np.matrix([[-m * l * g, 0]])]
    invs = [(np.matrix([[-1, 0], [1, 0], [0, -1], [0, 1]]),
        (np.matrix([[0.35], [0.35], [0.35], [0.35]])))]
    covers = [(invs[0][0], invs[0][1],
        np.matrix([[-0.35], [-0.35], [-0.35], [-0.35]]),
        np.matrix([[0.35], [0.35], [0.35], [0.35]]))]

    initial_shield = Shield(env, K_list=Ks, inv_list=invs, cover_list=covers,
            bound=episode_len)

    actor, shield = DDPG(env, args, rewardf=safety_reward,
            safe_training=safe_training,
            shields=shields, initial_shield=initial_shield, penalty_ratio=penalty_ratio)

    #################### Shield #################
    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    def rewardf(x, Q, u, R):
        return env.reward(x, u)

    if shield_test:
        shield.test_shield(actor, test_episodes, 3000, mode="single")

    actor.sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running Options')
    parser.add_argument('--nn_test', action="store_true", dest="nn_test")
    parser.add_argument('--retrain_shield', action="store_true",
            dest="retrain_shield")
    parser.add_argument('--shield_test', action="store_true",
            dest="shield_test")
    parser.add_argument('--test_episodes', action="store",
            dest="test_episodes", type=int)
    parser.add_argument('--retrain_nn', action="store_true", dest="retrain_nn")
    parser.add_argument('--safe_training', action="store_true",
            dest="safe_training")
    parser.add_argument('--shields', action="store", dest="shields", type=int)
    parser.add_argument('--episode_len', action="store", dest="ep_len", type=int)
    parser.add_argument('--max_episodes', action="store", dest="eps", type=int)
    parser.add_argument('--penalty_ratio', action="store", dest="ratio", type=float)
    parser_res = parser.parse_args()
    nn_test = parser_res.nn_test
    retrain_shield = parser_res.retrain_shield
    shield_test = parser_res.shield_test
    test_episodes = parser_res.test_episodes \
            if parser_res.test_episodes is not None else 100
    retrain_nn = parser_res.retrain_nn
    safe_training = parser_res.safe_training \
            if parser_res.safe_training is not None else False
    shields = parser_res.shields if parser_res.shields is not None else 1
    ep_len = parser_res.ep_len if parser_res.ep_len is not None else 50
    eps = parser_res.eps if parser_res.eps is not None else 1000
    ratio = parser_res.ratio if parser_res.ratio is not None else 0.1

    pendulum(eps, [1200,900], [1000,900,800],
            "ddpg_chkp/perfect_model/pendulum/change_l/", "random_search",
            100, 2000, nn_test=nn_test, retrain_shield=retrain_shield,
            shield_test=shield_test, test_episodes=test_episodes,
            retrain_nn=retrain_nn, safe_training=safe_training,
            shields=shields, episode_len=ep_len, penalty_ratio=ratio)
