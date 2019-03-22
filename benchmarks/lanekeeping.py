import sys
sys.path.append(".")

from main import *
from shield import Shield
from Environment import PolySysEnvironment
from DDPG import *
import argparse

def lanekeep (learning_method, number_of_rollouts, simulation_steps, learning_eposides, actor_structure, critic_structure, train_dir,\
            nn_test=False, retrain_shield=False, shield_test=False, test_episodes=100, retrain_nn=False):
  v0 = 27.7
  cf = 133000
  cr = 98800
  M  = 1650
  b  = 1.59
  a  = 1.11
  Iz = 2315.3

  ds = 4
  us = 2

  disturbance_x_min = np.array([[0],[0],[-0.035],[0]])
  disturbance_x_max = np.array([[0],[0],[ 0.035],[0]])

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    rd = random.uniform(-0.6, 0.6)
    delta = np.zeros((ds, 1), float)
    delta[0,0] = 1*x[1,0] + v0*x[2,0] + random.uniform(disturbance_x_min[0], disturbance_x_max[0])                                                                        #lateral displacement
    delta[1,0] = (-1*(cf+cr)/(M*v0))*x[1,0] + ((b*cr-a*cf)/(M*v0)-v0)*x[3,0] + (cf/M)*u[0,0] + random.uniform(disturbance_x_min[1], disturbance_x_max[1])                 #lateral velocity
    delta[2,0] = x[3,0] + random.uniform(disturbance_x_min[2], disturbance_x_max[2])                                                                                      #error yaw angle
    delta[3,0] = ((b*cr-a*cf)/(Iz*v0))*x[1,0] + (-1*(a*a*cf + b*b*cr)/(Iz*v0))*x[3,0] + (a*cf/Iz)*u[1,0]  + random.uniform(disturbance_x_min[3], disturbance_x_max[3])    #yaw rate

    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("1*x[2] + 27.7*x[3] + d[1]")
    f.append("(-1*(133000+98800)/(1650*27.7))*x[2] + ((1.59*98800-1.11*133000)/(1650*27.7)-27.7)*x[4] + (133000/1650)*{} + d[2]".format(kstr[0]))
    f.append("x[4] + d[3]")
    f.append("((1.59*98800-1.11*133000)/(2315.3*27.7))*x[2] + (-1*(1.11*1.11*133000 + 1.59*1.59*98800)/(2315.3*27.7))*x[4] + (1.11*133000/2315.3)*{} + d[4]".format(kstr[1]))
    return f

  h = 0.01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ -0.1],[ -0.1], [-0.1], [ -0.1]])
  s_max = np.array([[  0.1],[  0.1], [ 0.1], [  0.1]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005 0; 0 .0005")

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[0,0] > 0.9 or x[0, 0] < -0.9): # keep a safe distance from the car in front of you
      return True
    return False

  def unsafe_string():
    return ["-(x[1]- -0.9)*(0.9-x[1])"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))

    if (unsafe_eval(x)):
      reward -= 1e-3
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      return -1
    return 0 

  # Use sheild to directly learn a linear controller
  u_min = np.array([[-1]])
  u_max = np.array([[1]])
  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min = u_min, disturbance_x_min=disturbance_x_min, disturbance_x_max=disturbance_x_max, timestep=h)

  if retrain_nn:
    args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 1000,
         'max_episodes': 1000,
         'minibatch_size': 64,
         'random_seed': 2903,
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
         'max_episode_len': 1000,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 2903,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': nn_test, 
         'test_episodes': test_episodes,
         'test_episodes_len': 1000}

  actor =  DDPG(env, args)

  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'


  shield = Shield(env, actor, model_path=model_path, force_learning=retrain_shield)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag=0.4, step_size=0.5, without_nn_guide=True, aggressive=True)
  if shield_test:
    shield.test_shield(test_episodes, 1000, mode="single")

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

  lanekeep("random_search", 200, 200, 0, [240, 200], [280, 240, 200], "ddpg_chkp/lanekeeping/240200280240200/", \
    nn_test=nn_test, retrain_shield=retrain_shield, shield_test=shield_test, test_episodes=test_episodes, retrain_nn=retrain_nn)