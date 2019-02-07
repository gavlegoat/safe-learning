# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-11-10 16:53:08
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-11-10 21:35:58
# -------------------------------
from main import *

from shield import Shield
from Environment import Environment
from DDPG import *

# Show that there is an invariant that can prove the policy safe
def navigation (learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir, K=None):
  #Dynamics that are continuous!
  A = np.matrix([
    [ 0., 0., 1., 0.],
    [ 0., 0., 0., 1.],
    [ 0., 0., -1.2, .1],
    [ 0., 0., .1, -1.2]
    ])
  B = np.matrix([
    [0,0],[0,0],[1,0],[0,1]
    ])

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  target = np.array([[2.5],[4.5], [0], [0]])

  #intial state space
  s_min = np.array([[-3.5],[-3.5],[0],[0]])
  s_max = np.array([[-2.5],[-2.5],[0],[0]])

  #unsafe constraint
  unsafe_x_min = np.array([[-2],[-5], [np.NINF], [np.NINF]])
  unsafe_x_max = np.array([[ 5],[ 3], [np.inf], [np.inf]])

  s_min -= target
  s_max -= target
  unsafe_x_min = np.array([[-2-target[0,0]],[-5-target[1,0]], [np.NINF], [np.NINF]])
  unsafe_x_max = np.array([[ 5-target[0,0]],[ 3-target[1,0]], [np.inf], [np.inf]])

  #reward functions
  Q = np.zeros((4, 4), float)
  np.fill_diagonal(Q, 0.01)
  R = np.zeros((2,2), float)
  np.fill_diagonal(R, .0005)
  d, p = B.shape

  u_min = np.array([[-100], [-100]])
  u_max = np.array([[ 100], [ 100]])

  rewardf = lambda x, u: -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))

  env = Environment(A, B, u_min, u_max, s_min, s_max, unsafe_x_min, unsafe_x_max, Q, R, terminal_err=-0.1, continuous=True, multi_boundary=False, rewardf=rewardf, unsafe=True, bad_reward=-1000)

  ############ Train and Test NN model ############
  args = { 'actor_lr': 0.1,
         'critic_lr': 0.1,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 500,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': True, 
         'test_episodes': 10,
         'test_episodes_len': 2000}
  K = np.array([[ -2.757318,     8.14072623,  -3.40473968,   3.35465468],
                [ -5.38419094, -10.11003801,  -3.04579954,  -8.17257855]])
  replay_buffer = generate_replay_buffer_with_K(K, env, buffer_size=1000000, epsoides=2000,steps=500)
  actor =  DDPG(env, args, replay_buffer)

  #################### Shield #################
  def testf(x, u):
    unsafe = True
    for i in range(d):
      if unsafe_x_min[i, 0] != np.NINF:
        if not (unsafe_x_min[i, 0] <= x[i, 0] and x[i, 0] <= unsafe_x_max[i, 0]):
          unsafe = False
          break
    if (unsafe):
      print "unsafe : {}".format(x)
      return -1
    return 0  
  shield = Shield(env, None, train_dir+"K.model")
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, testf=testf, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5, bias=False)
  #shield.test_shield(1000, 5000)

  actor.sess.close()

#Correct K
K = np.array([[ -2.757318,     8.14072623,  -3.40473968,   3.35465468],
              [ -5.38419094, -10.11003801,  -3.04579954,  -8.17257855]])
#Error K
K = np.array([[ -5.8593484,   -3.44902618,  -6.41939682,  -5.95818385],
              [ -0.98328122, -13.18940486,   7.53294555,  -8.48188398]])
#K = [[-9.30297954,  8.74752577, -9.02805432,  0.91674678],
#    [-5.83383704, -8.22630367, -3.51927504, -7.28304629]]
if __name__ == "__main__":
  navigation("random_search", 100, 200, 10000, [300, 250, 200], [300, 250, 200, 150], "ddpg_chkp/car-navigation/1/300200300250200/")