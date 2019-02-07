from main import *

from shield import Shield
from Environment import PolySysEnvironment

from DDPG import *

# Show that there is an invariant that can prove the policy safe
def biology (learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  # 10-dimension and 1-input system and 1-disturbance system
  ds = 3
  us = 2

  #Dynamics that are defined as a continuous function!
  def f (x, u):
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
    if (x[0,0] >= 5):
      return True
    return False
  def unsafe_string():
    return ["x[1] - 5"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    if (unsafe_eval(x)):
      reward -= 100
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      print x
      return -1
    return 0 

  u_min = np.array([[-50.], [-50]])
  u_max = np.array([[ 50.], [ 50]])

  # K = np.array([[0, 0, 0],
  # [0,  0,  0]])

  # #sample an initial condition for system
  # x0 = np.matrix([
  #   [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #   [random.uniform(s_min[1, 0], s_max[1, 0])],
  #   [random.uniform(s_min[2, 0], s_max[2, 0])], 
  # ])

  # names = {0:"X", 1:"G", 2:"I"}

  # draw_controller_helper (f, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, timestep=h)

  ############ Train and Test NN model ############
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
       'enable_test': True, 
       'test_episodes': 100,
       'test_episodes_len': 1000}
  actor =  DDPG(env, args=args)
  #actor_boundary(env, actor, 1000, 100)

  #################### Shield #################
  model_path = os.path.split(args['model_path'])[0]+'/'
  linear_func_model_name = 'K.model'
  model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, actor, model_path=model_path, force_learning=False)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5)
  shield.test_shield(1000, 100, mode="single")
  shield.test_shield(1000, 100, mode="all")
  actor.sess.close()
  

biology ("random_search", 100, 300, 0, [240, 200], [280, 240, 200], "ddpg_chkp/biology/240200280240200/")