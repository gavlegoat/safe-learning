# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-11-03 19:02:00
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-11-04 18:57:37
# -------------------------------
from main import *
from Environment import Environment
from DDPG import *

def robot (learning_eposides, actor_structure, critic_structure, train_dir, learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([[0, 2],
    [1, 0]
    ])

  B = np.matrix([[1],
    [1]
    ])

  d, p = B.shape

  # the robot must move to a target region
  target_min = np.array([[ 8], [ 8]])
  target_max = np.array([[10], [10]])

  target = np.array([[0],[0]])
  for i in range(d):
    target[i, 0] = (target_min[i, 0] + target_max[i,0]) / 2
  print "target:\n {}".format(target)

  #intial state space
  s_min = np.array([[0.5],[0.5]])
  s_max = np.array([[1.5],[1.5]])

  u_min = np.array([[-2.]])
  u_max = np.array([[ 2.]])

  # Unsafe regions!
  unsafe_1_min = np.array([[np.NINF], [np.NINF]])
  unsafe_1_max = np.array([[-2], [np.inf]])

  unsafe_2_min = np.array([[10], [np.NINF]])
  unsafe_2_max = np.array([[np.inf], [np.inf]])

  unsafe_3_min = np.array([[np.NINF], [np.NINF]])
  unsafe_3_max = np.array([[np.inf], [-2]])

  unsafe_4_min = np.array([[np.NINF], [10]])
  unsafe_4_max = np.array([[np.inf], [np.inf]])

  unsafe_5_min = np.array([[4], [4]])
  unsafe_5_max = np.array([[8], [8]])

  min_avoid_array = np.array([unsafe_1_min, unsafe_2_min, unsafe_3_min, unsafe_4_min, unsafe_5_min])
  max_avoid_array = np.array([unsafe_1_max, unsafe_2_max, unsafe_3_max, unsafe_4_max, unsafe_5_max])

  Q = np.matrix("0.05 0 ; 0 0.05")
  R = np.matrix("0")

  def rewardf(x, u):
    return -np.dot((x-target).T,Q.dot(x-target))-np.dot(u.T,R.dot(u))

  env = Environment(A, B, u_min, u_max, s_min, s_max, min_avoid_array, max_avoid_array, Q, R, multi_boundary=True, rewardf=rewardf, unsafe=True, terminal_err=0.1, bad_reward=-100)

  # print replay_buffer.sample_batch(4)
  # print replay_buffer.sample_batch(4)
  # print replay_buffer.sample_batch(4)

  ############ Train and Test NN model ############
  args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 20,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': True, 
         'test_episodes': 10,
         'test_episodes_len': 20}
  #replay_buffer = random_search_for_init_buffer(env, args, target, trance_number=20, repeat_time=100, rewardf=rewardf, terminal_err=0.2)
  replay_buffer = None
  actor = DDPG(env, args, replay_buffer)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, explore_mag = 1.0, step_size = 1.0, bias=True)
  #shield.test_shield(1000, 5000)

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]
  robot(0, [300, 300, 200, 200], [400, 300, 300, 200], "ddpg_chkp/robot/1/discrete/300300200200400300300200/", "random_search", 200, 100, 10) 
