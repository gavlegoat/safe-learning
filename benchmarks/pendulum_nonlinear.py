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

def pendulum(learning_method, number_of_rollouts, simulation_steps,
        learning_episodes, critic_structure, actor_structure, train_dir,
        nn_test=False, retrain_shield=False, shield_test=False,
        test_episodes=100, retrain_nn=False, safe_training=False, shields=1,
        episode_len=100, penalty_ratio=0.1):

    # Two dimensional state space: angle and angular velocity. We will assume that
    # downward is 0, and the goal is to reach pi or -pi
    # One dimensional action space: torque to apply
    # The safety condition here will be a bound on the angular velocity of the
    # pendulum.

    l = 4.0
    t = 10.0
    max_speed = 1.5

    def f(x, u):
        # Pendulums are governed by d^2 theta / d t^2 + g / l sin(theta) = 0, so we have
        # omega += u - g / l sin(theta), r += omega
        return np.matrix([[x[1,0]],
                          [t * u[0,0] - 9.81 / l * np.sin(x[0,0])],
                          [0.0]])

    def f_to_str(K):
        raise NotImplementedError

    def rewardf(x, Q, u, R):
        return max(-np.abs(x[0,0] - np.pi), -np.abs(x[0,0] + np.pi))

    def testf(x, u):
        return np.abs(x[1,0]) >= max_speed

    x_min = np.array([[-3.3], [-5.0], [1.0]])
    x_max = np.array([[ 3.3], [ 5.0], [1.0]])

    s_min = np.array([[0.0], [0.0], [1.0]])
    s_max = np.array([[0.0], [0.0], [1.0]])

    u_min = np.array([[-2.0]])
    u_max = np.array([[ 2.0]])

    # sin on -pi to pi, we'll have a ramp and a plateau piece:
    breaks = [-3 * np.pi / 2 + 0.6, -np.pi / 2 - 0.6, -np.pi / 2 + 0.6,
              np.pi / 2 - 0.6, np.pi / 2 + 0.6, 3 * np.pi / 2 - 0.6]
    break_breaks = [6, 6, 6]
    mins = [-1, -np.sin(-np.pi / 2 - 0.6), -np.sin(-np.pi / 2 + 0.6), -1, -1, -np.sin(3 * np.pi / 2 - 0.6)]
    maxes = [-np.sin(-3 * np.pi / 2 + 0.6), 1, 1, -np.sin(np.pi / 2 - 0.6), -np.sin(np.pi / 2 + 0.6), 1]
    lower_As = []
    upper_As = []
    B = np.array([[0.0], [t], [0.0]])
    for i in range(len(breaks) - 1):
        max_m = (maxes[i+1] - maxes[i]) / (breaks[i+1] - breaks[i])
        min_m = (mins[i+1] - mins[i]) / (breaks[i+1] - breaks[i])
        # lA * x + B * u <= x' <= uA * x + B * u
        lower_As.append(np.matrix([[0.0, 1.0], [9.81 / l * min_m, 0.0], [0.0, 0.0]]))
        upper_As.append(np.matrix([[0.0, 1.0], [9.81 / l * max_m, 0.0], [0.0, 0.0]]))
    Bs = [B] * len(lower_As)

    uA = np.matrix([[0.0, -1.0, 0.0]])
    ub = np.matrix([[-max_speed]])

    env = PolySysEnvironment(f, f_to_str, rewardf, testf, None, 3, 1,
            None, None, s_min, s_max, x_min=x_min, x_max=x_max,
            u_min=u_min, u_max=u_max, timestep=0.01,
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

    Ks = [np.matrix([[0.0, 0.0, 0.0]])]
    invs = [(np.matrix([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]),
        np.matrix([[3.3], [3.3], [max_speed], [max_speed]]))]
    covers = [(invs[0][0], invs[0][1], np.matrix([[-2.0], [-max_speed / 2], [1.0]]),
        np.matrix([[2.0], [max_speed / 2], [1.0]]))]

    bound = 10

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

    pendulum("random_search", 200, 100, eps, [240, 200], [280, 240, 200],
            "ddpg_chkp/pendulum/240200280240200/",
            nn_test=nn_test, retrain_shield=retrain_shield,
            shield_test=shield_test, test_episodes=test_episodes,
            retrain_nn=retrain_nn, safe_training=safe_training,
            shields=shields, episode_len=ep_len, penalty_ratio=ratio)
