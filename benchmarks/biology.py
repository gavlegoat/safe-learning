import sys
sys.path.append(".")

from main import *
from shield import Shield
from Environment import PolySysEnvironment
from DDPG import *
import argparse
import synthesis

def biology (learning_method, number_of_rollouts, simulation_steps,
        learning_eposides, critic_structure, actor_structure, train_dir,
        nn_test=False, retrain_shield=False, shield_test=False,
        test_episodes=100, retrain_nn=False, safe_training=False, shields=1,
        episode_len=100):

    # 10-dimension and 1-input system and 1-disturbance system
    ds = 3
    us = 2

    #Dynamics that are defined as a continuous function!
    def f(x, u):
        #random disturbance
        #d = random.uniform(0, 20)
        delta = np.zeros((ds, 1), float)
        delta[0,0] = -0.01*x[0,0] - x[1,0]*(x[0,0]+4.5) + u[0,0]
        delta[1,0] = -0.025*x[1,0] + 0.000013*x[2,0]
        delta[2,0] = -0.093*(x[2,0] + 15) + (1/12)*u[1,0]
        return delta

    #Closed loop system dynamics to text
    def f_to_str(K):
        kstr = K_to_str(K)
        f = []
        f.append("-0.01*x[1] - x[2]*(x[1]+4.5) + {}".format(kstr[0]))
        f.append("-0.025*x[2] + 0.000013*x[3]")
        f.append("-0.093*(x[3] + 15) + (1/12)*{}".format(kstr[1]))
        return f

    h = 0.01

    # amount of Gaussian noise in dynamics
    eq_err = 1e-2

    #intial state space
    s_min = np.array([[-2],[-0],[-0.1]])
    s_max = np.array([[ 2],[ 0],[ 0.1]])

    Q = np.zeros((ds,ds), float)
    R = np.zeros((us,us), float)
    np.fill_diagonal(Q, 1)
    np.fill_diagonal(R, 1)

    #user defined unsafety condition
    def unsafe_eval(x):
        if x[0,0] >= 5:
            return True
        return False
    def unsafe_string():
        return ["x[1] - 5"]

    def safety_reward(x, Q, u, R):
        return -np.matrix([[x[0,0]]])

    def rewardf(x, Q, u, R):
        reward = 0
        reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
        if unsafe_eval(x):
            reward -= 100
        return reward

    def testf(x, u):
        if unsafe_eval(x):
            print x
            return -1
        return 0

    u_min = np.array([[-50.], [-50]])
    u_max = np.array([[ 50.], [ 50]])

    env_capsule = synthesis.get_env_capsule("biology")

    unsafe_A = [np.matrix([[-1, 0, 0]])]
    unsafe_b = [np.matrix([[5]])]

    env = PolySysEnvironment(f, f_to_str, rewardf, testf, unsafe_string, ds,
            us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, timestep=h,
            capsule=env_capsule, unsafe_A=unsafe_A, unsafe_b=unsafe_b)

    ############ Train and Test NN model ############
    if retrain_nn:
        args = { 'actor_lr': 0.0001,
                 'critic_lr': 0.001,
                 'actor_structure': actor_structure,
                 'critic_structure': critic_structure,
                 'buffer_size': 1000000,
                 'gamma': 0.99,
                 'max_episode_len': episode_len,
                 'max_episodes': 1000,
                 'minibatch_size': 64,
                 'random_seed': 6553,
                 'tau': 0.005,
                 'model_path': train_dir+"retrained_model.chkp",
                 'enable_test': nn_test,
                 'test_episodes': test_episodes,
                 'test_episodes_len': 1000}
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
                 'test_episodes_len': 1000}

    # We just need to keep x1 small, so if x1 is positive we will make the
    # action be large and negative, otherwise we do nothing.
    Ks = [np.matrix([[-100, 0, 0], [0, 0, 0]]),
            np.matrix([[0, 0, 0], [0, 0, 0]])]
    invs = [(np.matrix([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]]),
        np.matrix([[5], [0], [5], [5], [5], [5]])),
       (np.matrix([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]]),
        np.matrix([[0], [5], [5], [5], [5], [5]]))]
    covers = [(invs[0][0], invs[0][1],
        np.matrix([[0], [-0.1], [-0.1]]),
        np.matrix([[2], [0.1], [0.1]])),
        (invs[1][0], invs[1][1],
            np.matrix([[-2], [-0.1], [-0.1]]),
            np.matrix([[0], [0.1], [0.1]]))]

    initial_shield = Shield(env, K_list=Ks, inv_list=invs, cover_list=covers,
            bound=episode_len)
    actor, shield = DDPG(env, args=args, rewardf=safety_reward,
            safe_training=safe_training, shields=shields,
            initial_shield=initial_shield)

    #################### Shield #################
    model_path = os.path.split(args['model_path'])[0]+'/'
    linear_func_model_name = 'K.model'
    model_path = model_path+linear_func_model_name+'.npy'

    #shield = Shield(env, actor, model_path=model_path,
    #        force_learning=retrain_shield)
    #shield.train_polysys_shield(learning_method, number_of_rollouts,
    #        simulation_steps, eq_err=eq_err, explore_mag=0.4, step_size=0.5,
    #        aggressive=True, without_nn_guide=True, enable_jit=True)
    if shield_test:
        shield.test_shield(actor, test_episodes, 1000, mode="single")
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

    biology ("random_search", 200, 500, 0, [240, 200], [280, 240, 200],
            "ddpg_chkp/biology/240200280240200/", nn_test=nn_test,
            retrain_shield=retrain_shield, shield_test=shield_test,
            test_episodes=test_episodes, retrain_nn=retrain_nn,
            safe_training=safe_training, shields=shields,
            episode_len=ep_len)
