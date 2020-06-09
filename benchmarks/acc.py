import sys
sys.path.append(".")

import numpy as np
from DDPG import *
from main import *
import os.path
from Environment import Environment, PolySysEnvironment
from shield import Shield
import argparse
import random

def acc(learning_method, number_of_rollouts, simulation_steps,
        learning_episodes, actor_structure, critic_structure, train_dir,
        nn_test=False, retrain_shield=False, shield_test=False,
        test_episodes=100, retrain_nn=False, safe_training=False, shields=1,
        episode_len=100, penalty_ratio=0.1):

    # Adaptive cruise control: We use a relavice reference frame in which the
    # ego car gets a reward for being close to the lead car.
    # x0 is the negative distance between the cars
    # x1 is their relative speeds (positive if the ego car is catching up)
    # u1 is the acceleration of the ego car

    a_min = -3
    a_max = 2

    def f(x, u):
        #lead_a = random.random() * (a_max - a_min) + a_min
        lead_a = max(a_min, min(a_max, random.gauss(0, 1)))
        return np.matrix([[x[1,0]],
                          [u[0,0] - lead_a],
                          [0.0]])

    def f_to_str(K):
        raise NotImplementedError

    def rewardf(x, Q, u, R):
        return x[0,0]

    def testf(x, u):
        return x[0,0] >= 0

    def terminalf(x):
        # We don't terminate episodes early unless we hit a bad state
        return False

    x_min = np.array([[-5.0], [-5.0], [1.0]])
    x_max = np.array([[ 0.5], [5.0], [1.0]])

    s_min = np.array([[-1.1], [-0.1], [1.0]])
    s_max = np.array([[-0.9], [ 0.1], [1.0]])

    u_min = np.array([[a_min]])
    u_max = np.array([[a_max]])

    B = np.matrix([[0.0, 1.0, 0.0]])
    lower_A = np.matrix([[0.0, 1.0, 0.0],
                         [0.0, 0.0, -a_max],
                         [0.0, 0.0, 0.0]])
    upper_A = np.matrix([[0.0, 1.0, 0.0],
                         [0.0, 0.0, -a_min],
                         [0.0, 0.0, 0.0]])

    breaks = []
    break_breaks = []

    uA = np.matrix([[-1.0, 0.0, 0.0]])
    ub = np.matrix([[0.0]])

    env = PolySysEnvironment(f, f_to_str, rewardf, testf, None, 3, 1,
            None, None, s_min, s_max, x_min=x_min, x_max=x_max,
            u_min=u_min, u_max=u_max, timestep=0.01,
            unsafe_A=uA, unsafe_b=ub, approx=True, breaks=breaks,
            break_breaks=break_breaks,
            lower_As=[lower_A], lower_Bs=[B],
            upper_As=[upper_A], upper_Bs=[B],
            terminalf=terminalf);

    if retrain_nn:
        args = { 'actor_lr': 0.0001,    # [240, 200]
                 'critic_lr': 0.001,    # [280, 240, 200]
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure, 
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,   # 100
                 'max_episodes': learning_episodes,   # originally 1000
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"retrained_model.chkp",
                 'enable_test': nn_test, 
                 'test_episodes': test_episodes,
                 'test_episodes_len': 5000}
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
                 'test_episodes_len': 5000}

    # Initial strategy: Always brake hard
    Ks = [np.matrix([[0.0, 0.0, a_min]])]
    invs = [(np.matrix([[1.0, 0.0, 0.0]]), np.matrix([[-0.1]]))]
    covers = [(invs[0][0], invs[0][1], np.matrix([[-2.0], [-2.0], [1.0]]),
        np.matrix([[-0.5], [2.0], [1.0]]))]

    bound = 30

    initial_shield = Shield(env, K_list=Ks, inv_list=invs, cover_list=covers,
            bound=bound)

    actor, shield = DDPG(env, args, safe_training=safe_training, shields=shields,
            initial_shield=initial_shield, penalty_ratio=penalty_ratio, bound=bound)

    if shield_test:
        shield.test_shield(actor, test_episodes, 5000)

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

    acc("random_search", 200, 100, eps, [240, 200], [280, 240, 200],
            "ddpg_chkp/acc/240200280240200/",
            nn_test=nn_test, retrain_shield=retrain_shield,
            shield_test=shield_test, test_episodes=test_episodes,
            retrain_nn=retrain_nn, safe_training=safe_training,
            shields=shields, episode_len=ep_len, penalty_ratio=ratio)
