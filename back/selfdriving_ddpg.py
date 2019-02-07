# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-11-06 12:23:39
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-11-06 16:46:13
# -------------------------------
from main import *

from shield import Shield
from Environment import PolySysEnvironment

from DDPG import *

# Show that there is an invariant that can prove the policy safe
def selfdrive(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir, K=None):
  # 2-dimension and 1-input system
  ds = 2
  us = 1

  #the speed is set to 2 in this case
  v = 2
  cl = 2
  cr = -2

  #Dynamics that are defined as a continuous function!
  # def f (x, u):
  #   #We have two aeroplanes with 2 inputs for each controlling its own angular velocity!
  #   delta = np.zeros((ds, 1), float)
  #   delta[0, 0] = -v*math.sin(x[1, 0])      #distance
  #   delta[1, 0] = u[0, 0]                   #angular velocity (controlled by AI)
  #   return delta

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

  args = { 'actor_lr': 0.0001,
       'critic_lr': 0.001,
       'actor_structure': actor_structure,
       'critic_structure': critic_structure, 
       'buffer_size': 1000000,
       'gamma': 0.99,
       'max_episode_len': 1,
       'max_episodes': learning_eposides,
       'minibatch_size': 64,
       'random_seed': 6553,
       'tau': 0.005,
       'model_path': train_dir+"model.chkp",
       'enable_test': False, 
       'test_episodes': 10,
       'test_episodes_len': 5000}
  actor =  DDPG(env, args=args)

  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path=model_path, force_learning=True)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps)
  shield.test_shield(10, 5000)

  actor.sess.close()

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  selfdrive("random_search", 100, 100, 0, [300, 200], [300, 250, 200], "ddpg_chkp/selfdriving/300200300250200")