import sys
sys.path.append(".")

from main import *
from shield import Shield
from Environment import PolySysEnvironment
from DDPG import *
import argparse

# Show that there is an invariant that can prove the policy safe
def selfdrive(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir, \
            nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100, retrain_nn=False):
  # 2-dimension and 1-input system
  ds = 2
  us = 1

  #the speed is set to 2 in this case
  v = 2
  cl = 2
  cr = -2

  def f(x, u):
    #We have two aeroplanes with 2 inputs for each controlling its own angular velocity!
    delta = np.zeros((ds, 1), float)
    delta[0, 0] = -v*(x[1,0] - ((pow(x[1,0],3))/6))
    delta[1, 0] = u[0, 0]                   #angular velocity (controlled by AIs)
    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("-{}*(x[2] - ((x[2]^3)/6))".format(v))
    f.append(kstr[0])
    return f

  h = 0.1

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  pi = 3.1415926

  #intial state space
  s_min = np.array([[-1],[-pi/4]])
  s_max = np.array([[ 1],[ pi/4]])

  u_min = np.array([[-10]])
  u_max = np.array([[10]])

  #the only portion of the entire state space that our verification is interested.
  bound_x_min = np.array([[None],[-pi/2]])
  bound_x_max = np.array([[None],[ pi/2]])

  #sample an initial condition for system
  x0 = np.matrix([
    [random.uniform(s_min[0, 0], s_max[0, 0])], 
    [random.uniform(s_min[1, 0], s_max[1, 0])]
  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  #reward functions
  Q = np.zeros((2,2), float)
  np.fill_diagonal(Q, 1)
  R = np.zeros((1,1), float)
  np.fill_diagonal(R, 1)

  #user defined unsafety condition
  def unsafe_eval(x):
    outbound1 = -(x[0,0]- cr)*(cl-x[0,0])
    if (outbound1 >= 0):
      return True
    return False
  def unsafe_string():
    return ["-(x[1]- {})*({}-x[1])".format(cr, cl)]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x)) -np.dot(u.T,R.dot(u))
    if (unsafe_eval(x)):
      reward -= 100
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      return -1
    return 0 

  def random_test(f, K, simulation_steps, continuous=True, timestep=h):
    total_fails = 0
    for i in range(100):
      x0 = np.matrix([
        [random.uniform(s_min[0, 0], s_max[0, 0])], 
        [random.uniform(s_min[1, 0], s_max[1, 0])]
      ])
      reward = test_controller_helper (f, K, x0, simulation_steps, testf, continuous=True, timestep=h)
      if reward < 0:
        print "Failed on {}".format(x0)
        total_fails += 1
    print ("Among {} tests {} are failed.".format(100, total_fails))

  names = {0:"p", 1:"gamma"}

  # Use sheild to directly learn a linear controller
  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, bound_x_min=bound_x_min, bound_x_max=bound_x_max, timestep=0.1)

  if retrain_nn:
    args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 100,
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
         'max_episode_len': 100,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': nn_test, 
         'test_episodes': test_episodes,
         'test_episodes_len': 1000}
  actor =  DDPG(env, args=args)

  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  def rewardf(x, Q, u, R):
    return np.matrix([[env.reward(x, u)]])

  shield = Shield(env, actor, model_path=model_path, force_learning=retrain_shield)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, explore_mag = 0.04, step_size = 0.03, without_nn_guide=True)
  if shield_test:
    shield.test_shield(test_episodes, 1000)

  actor.sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Running Options')
  parser.add_argument('--nn_test', action="store_true", dest="nn_test")
  parser.add_argument('--retrain_shield', action="store_true", dest="retrain_shield")
  parser.add_argument('--shield_test', action="store_true", dest="shield_test")
  parser.add_argument('--test_episodes', action="store", dest="test_episodes", type=int)
  parser.add_argument('--retrain_nn', action="store_true", dest="retrain_nn")
  parser_res = parser.parse_args()
  nn_test = parser_res.nn_test
  retrain_shield = parser_res.retrain_shield
  shield_test = parser_res.shield_test
  test_episodes = parser_res.test_episodes if parser_res.test_episodes is not None else 100
  retrain_nn = parser_res.retrain_nn

  selfdrive("random_search", 200, 200, 0, [300, 200], [300, 250, 200], "ddpg_chkp/selfdriving/300200300250200/", \
    nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes, retrain_nn=retrain_nn)