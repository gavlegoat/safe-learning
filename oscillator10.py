from main import *

from shield import Shield
from Environment import PolySysEnvironment

from DDPG import *

# Show that there is an invariant that can prove the policy safe
def oscillator(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir):
  # 10-dimension and 1-input system and 1-disturbance system
  ds = 10
  us = 2

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    #random disturbance
    #d = random.uniform(0, 20)
    delta = np.zeros((ds, 1), float)
    delta[0,0] = -2*x[0,0] +u[0,0]
    delta[1,0] = -x[1,0] + u[1,0]
    delta[2,0] = 5*x[0,0] - 5*x[2,0]
    delta[3,0] = 5*x[2,0] - 5*x[3,0]
    delta[4,0] = 5*x[3,0] - 5*x[4,0]
    delta[5,0] = 5*x[4,0] - 5*x[5,0]
    delta[6,0] = 5*x[5,0] - 5*x[6,0]
    delta[7,0] = 5*x[6,0] - 5*x[7,0]
    delta[8,0] = 5*x[7,0] - 5*x[8,0]
    delta[9,0] = 5*x[8,0] - 5*x[9,0]
    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("-2*x[1] + {}".format(kstr[0]))
    f.append("-x[2] + {}".format(kstr[1]))
    f.append("5*x[1]-5*x[3]")
    f.append("5*x[3]-5*x[4]")
    f.append("5*x[4]-5*x[5]")
    f.append("5*x[5]-5*x[6]")
    f.append("5*x[6]-5*x[7]")
    f.append("5*x[7]-5*x[8]")
    f.append("5*x[8]-5*x[9]")
    f.append("5*x[9]-5*x[10]")
    return f

  h = 0.01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[0.2],[-0.1], [0], [0], [0], [0], [0], [0], [0], [0]])
  s_max = np.array([[0.3],[ 0.1], [0], [0], [0], [0], [0], [0], [0], [0]])


  Q = np.zeros((ds,ds), float)
  R = np.zeros((us,us), float)
  np.fill_diagonal(Q, 1)
  np.fill_diagonal(R, 1)

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[9,0] >= 0.05):
      return True
    return False
  def unsafe_string():
    return ["x[10] - 0.05"]

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

  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, timestep=h)


  ############ Train and Test NN model ############
  args = { 'actor_lr': 0.0001,
       'critic_lr': 0.001,
       'actor_structure': actor_structure,
       'critic_structure': critic_structure, 
       'buffer_size': 1000000,
       'gamma': 0.99,
       'max_episode_len':64,
       'max_episodes': learning_eposides,
       'minibatch_size': 20,
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
  shield.test_shield(100, 1000, mode="single")
  #shield.test_shield(100, 1000, mode="all")
  actor.sess.close()
  
oscillator("random_search", 200, 200, 0, [240, 200], [280, 240, 200], "ddpg_chkp/oscillator/10/240200280240200/")