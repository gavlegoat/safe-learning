import sys
sys.path.append(".")

import numpy as np
from DDPG import *
from main import *
import os.path
from Environment import Environment, PolySysEnvironment
from shield import Shield
import argparse

def road(learning_method, number_of_rollouts, simulation_steps,
        learning_episodes, actor_structure, critic_structure, train_dir,
        nn_test=False, retrain_shield=False, shield_test=False,
        test_episodes=100, retrain_nn=False, safe_training=False, shields=1,
        episode_len=100, penalty_ratio=0.1):

    # States: pos_x, pos_y, vel_x, vel_y, constant
    # Actions: acc_x, acc_y

    A = np.matrix([[0.0, 0.0, 10.0,  0.0, 0.0],
                   [0.0, 0.0,  0.0, 10.0, 0.0],
                   [0.0, 0.0,  0.0,  0.0, 0.0],
                   [0.0, 0.0,  0.0,  0.0, 0.0],
                   [0.0, 0.0,  0.0,  0.0, 0.0]])
    B = 0.5 * np.matrix([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])

    #s_min = np.array([[-0.1], [-0.1], [-0.1], [-0.1], [1]])
    #s_max = np.array([[0.1], [0.1], [0.1], [0.1], [1]])
    s_min = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    s_max = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])

    x_goal = 3.0
    y_goal = 3.0

    max_speed = 10.0

    x_min = np.array([[-100.0], [-100.0], [-max_speed], [-max_speed], [0.0]])
    x_max = np.array([[ 100.0], [ 100.0], [ max_speed], [ max_speed], [2.0]])

    def f(x, u):
        dA = -0.06 + 0.12 * np.random.rand(5, 5)
        for i in range(5):
            dA[4,i] = 0.0
        dB = -0.06 + 0.12 * np.random.rand(5, 2)
        for i in range(2):
            dB[4,i] = 0.0
        return (A + dA) * x + (B + dB) * u

    def f_to_str(K):
        raise NotImplementedError

    def testf(x, u):
        return np.sqrt(x[2,0]**2 + x[3,0]**2) > max_speed

    def rewardf(x, Q, u, R):
        return -(abs(x[0,0] - x_goal) + abs(x[1,0] - y_goal))

    def terminalf(x):
        return x[0,0] >= x_goal and x[1,0] >= y_goal

    lower_As = [A - np.full(A.shape, 0.06)]
    upper_As = [A + np.full(A.shape, 0.06)]
    lower_Bs = [B - np.full(B.shape, 0.06)]
    upper_Bs = [B + np.full(B.shape, 0.06)]

    u_min = np.array([[-2.0], [-2.0]])
    u_max = np.array([[ 5.0], [ 5.0]])

    # The safety property here is that sqrt(v_x^2 + v_y^2) < max_speed, but we
    # can't handle this constraint so we need a linear overapproximation. We
    # can do this by circumscribing an n-gon within the circle defined by the
    # actual safety constraint
    n = 8
    xs = []
    ys = []
    for i in range(n):
        ang = 2 * i * np.pi / n + np.pi / n
        xs.append(max_speed * np.cos(ang))
        ys.append(max_speed * np.sin(ang))
    unsafe_A = []
    unsafe_b = []
    for i in range(n):
        j = (i + 1) % n
        if abs(xs[j] - xs[i]) < 0.000001:
            if xs[i] >= 0.0:
                unsafe_A.append(np.matrix([[0.0, 0.0, -1.0, 0.0, 0.0]]))
                unsafe_b.append(np.matrix([[-xs[i]]]))
            else:
                unsafe_A.append(np.matrix([[0.0, 0.0, 1.0, 0.0, 0.0]]))
                unsafe_b.append(np.matrix([[xs[i]]]))
        else:
            m = (ys[j] - ys[i]) / (xs[j] - xs[i])
            b = ys[i] - m * xs[i]
            if b >= 0.0:
                unsafe_A.append(np.matrix([[0.0, 0.0, m, -1.0, 0.0]]))
                unsafe_b.append(np.matrix([[-b]]))
            else:
                unsafe_A.append(np.matrix([[0.0, 0.0, -m, 1.0, 0.0]]))
                unsafe_b.append(np.matrix([[b]]))

    breaks = []
    break_breaks = []

    env = PolySysEnvironment(f, f_to_str, rewardf, testf, None, 5, 2,
            None, None, s_min, s_max, x_min=x_min, x_max=x_max,
            u_min=u_min, u_max=u_max,
            unsafe_A=unsafe_A, unsafe_b=unsafe_b, timestep=0.01,
            approx=True, breaks=breaks, break_breaks=[],
            lower_As=lower_As, lower_Bs=lower_Bs,
            upper_As=upper_As, upper_Bs=upper_Bs,
            terminalf=terminalf)

    if retrain_nn:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure, 
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,
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
                 'max_episodes': learning_episodes,
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"model.chkp",
                 'enable_test': nn_test, 
                 'test_episodes': test_episodes,
                 'test_episodes_len': 5000}

    Ks = [np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]])]
    inv_A = np.matrix(np.zeros((len(unsafe_A), unsafe_A[0].shape[1])))
    inv_b = np.matrix(np.zeros((len(unsafe_b), 1)))
    for (i, (A, b)) in enumerate(zip(unsafe_A, unsafe_b)):
        inv_A[i] = -unsafe_A[i][0]
        inv_b[i,0] = -2.0 * unsafe_b[i][0,0]
    invs = [(inv_A, inv_b)]
    covers = [(invs[0][0], invs[0][1], np.matrix([[-1.0, -1.0, -1.0, -1.0, 1.0]]),
            np.matrix([[x_goal, y_goal, max_speed / 2.0, max_speed / 2.0, 1.0]]))]

    bound = 20

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

    road("random_search", 200, 100, eps, [240, 200], [280, 240, 200],
            "ddpg_chkp/road/240200280240200/",
            nn_test=nn_test, retrain_shield=retrain_shield,
            shield_test=shield_test, test_episodes=test_episodes,
            retrain_nn=retrain_nn, safe_training=safe_training,
            shields=shields, episode_len=ep_len, penalty_ratio=ratio)
