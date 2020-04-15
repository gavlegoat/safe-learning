import sys
sys.path.append(".")

import numpy as np
from DDPG import *
from main import *
import os.path
from Environment import Environment, PolySysEnvironment
from shield import Shield
import argparse

import synthesis

def car(learning_method, number_of_rollouts, simulation_steps,
        learning_episodes, actor_structure, critic_structure, train_dir,
        nn_test=False, retrain_shield=False, shield_test=False,
        test_episodes=100, retrain_nn=False, safe_training=False, shields=1,
        episode_len=100, penalty_ratio=0.1):

    # Set up the actual model
    def f(x, u):
        return np.matrix([[x[1,0]], [0.001 * u[0,0] - 0.0025 * np.cos(3 * x[0,0])]])

    def f_to_str(K):
        kstr = K_to_str(K)
        f = []
        f.append("x[2]")
        f.append("0.001*{} - 0.0025 * cos(3 * x[1])".format(kstr[0]))
        return f

    def rewardf(x, Q, u, R):
        return x[0,0] - 0.6

    def testf(x, u):
        return x[0,0] < -np.pi / 3

    x_min = np.array([[-1.2], [-0.007]])
    x_max = np.array([[0.7], [0.007]])

    s_min = np.array([[-0.5], [0.0]])
    s_max = np.array([[-0.5], [0.0]])

    u_min = np.array([[-1.0]])
    u_max = np.array([[1.0]])

    # Set up a linearized model
    # We'll use splits at -0.1 to 0.1 around each peak (since the dynamics
    # use cos(3x), the peaks are at multiples of pi / 3).
    breaks = [-np.pi / 3 - 0.2, -np.pi / 3 + 0.2, -0.2, 0.2, np.pi / 3 - 0.2]
    break_breaks = [5, 5, 5]
    mins = [-np.cos(-np.pi - 0.6), -np.cos(-np.pi + 0.6), -1, -1, np.cos(np.pi - 0.6)]
    maxes = [1, 1, -np.cos(-0.6), -np.cos(0.6), 1]
    lower_As = []
    upper_As = []
    B = np.array([[0.0], [0.001]])
    for i in range(len(breaks) - 1):
        max_m = (maxes[i+1] - maxes[i]) / (breaks[i+1] - breaks[i])
        min_m = (mins[i+1] - mins[i]) / (breaks[i+1] - breaks[i])
        # lA * x + B * u <= x' <= uA * x + B * u
        lower_As.append(np.matrix([[0.0, 1.0], [0.0025 * min_m, 0.0]]))
        upper_As.append(np.matrix([[0.0, 1.0], [0.0025 * max_m, 0.0]]))
    Bs = [B] * len(lower_As)

    # We consider unsafe behavior to be moving over the left side of the hill
    uA = np.matrix([[1.0, 0.0]])
    ub = np.matrix([[-np.pi / 3]])

    env = PolySysEnvironment(f, f_to_str, rewardf, testf, None, 2, 1,
            None, None, s_min, s_max, x_min=x_min, x_max=x_max,
            u_min=u_min, u_max=u_max, timestep=1.0,
            unsafe_A=uA, unsafe_b=ub, approx=True, breaks=breaks,
            break_breaks=break_breaks,
            lower_As=lower_As, lower_Bs=Bs, upper_As=upper_As, upper_Bs=Bs)

    if retrain_nn:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure,
                 'buffer_size': 1000000,
                 'gamma': 0.99, 'max_episode_len': episode_len,
                 'max_episodes': learning_episodes,
                 'minibatch_size': 64,
                 'random_seed': 6554,   # 6553
                 'tau': 0.005,
                 'model_path': train_dir+"retrained_model.chkp",
                 'enable_test': nn_test,
                 'test_episodes': test_episodes,
                 'test_episodes_len': 500}
    else:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure,
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,
                 'max_episodes': 0,
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"model.chkp",
                 'enable_test': nn_test,
                 'test_episodes': test_episodes,
                 'test_episodes_len': 500}

    Ks = [np.matrix([[0.0, 0.0]])]
    invs = [(np.matrix([[1, 0], [-1, 0], [0, 1], [0, -1]]),
        np.matrix([[0.8], [0.8], [0.07], [0.07]]))]
    covers = [(invs[0][0], invs[0][1], np.matrix([[-0.8], [-0.07]]),
        np.matrix([[0.8], [0.07]]))]

    bound = 30
    initial_shield = Shield(env, K_list=Ks, inv_list=invs, cover_list=covers,
            bound=bound)

    actor, shield = DDPG(env, args,
            safe_training=safe_training,
            shields=shields, initial_shield=initial_shield,
            penalty_ratio=penalty_ratio)

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

    car("random_search", 200, 100, eps, [240, 200], [280, 240, 200],
            "ddpg_chkp/road/240200280240200/",
            nn_test=nn_test, retrain_shield=retrain_shield,
            shield_test=shield_test, test_episodes=test_episodes,
            retrain_nn=retrain_nn, safe_training=safe_training,
            shields=shields, episode_len=ep_len, penalty_ratio=ratio)
