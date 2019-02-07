from main import *

from shield import Shield
from Environment import PolySysEnvironment

#from DDPG import *

# A nonlinear oscillator of second order.

# Show that there is an invariant that can prove the policy safe
def oscillator(learning_method, number_of_rollouts, simulation_steps, learning_eposides, critic_structure, actor_structure, train_dir, K=None):
  # 2-dimension and 1-input system
  ds = 2
  us = 1

  def f(x, u):
    #We have two aeroplanes with 2 inputs for each controlling its own angular velocity!
    delta = np.zeros((ds, 1), float)
    delta[0, 0] = x[1,0] 
    delta[1, 0] = -0.6*x[1,0] - x[0,0] - x[0,0]*x[0,0]*x[0,0] + u[0, 0]
    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("x[2]")
    f.append("-0.6*x[2] - x[1] - x[1]*x[1]*x[1] + {}".format(kstr[0]))
    return f

  # amount of Gaussian noise in dynamics
  eq_err = 0

  #intial state space
  s_min = np.array([[-2.5],[-2]])
  s_max = np.array([[ 2.5],[ 2]])

  u_min = np.array([[-10]])
  u_max = np.array([[10]])

  #the only portion of the entire state space that our verification is interested.
  bound_x_min = np.array([[-5],[-5]])
  bound_x_max = np.array([[ 5],[ 5]]) 

  #reward functions
  Q = np.zeros((2,2), float)
  np.fill_diagonal(Q, 1)
  R = np.zeros((1,1), float)
  np.fill_diagonal(R, 0.1)

  #user defined unsafety condition
  def unsafe_eval(x):
    outbound1 = -(x[1,0]- -4)*(4-x[1,0])
    if (outbound1 >= 0):
      return True
    return False
  def unsafe_string():
    return ["-(x[2]- {})*({}-x[2])".format(-4, 4)]

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

  # Use sheild to directly learn a linear controller
  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, u_max=u_max, u_min=u_min, bound_x_min=bound_x_min, bound_x_max=bound_x_max, timestep=0.01)

  # args = { 'actor_lr': 0.0001,
  #      'critic_lr': 0.001,
  #      'actor_structure': actor_structure,
  #      'critic_structure': critic_structure, 
  #      'buffer_size': 1000000,
  #      'gamma': 0.99,
  #      'max_episode_len': 1,
  #      'max_episodes': learning_eposides,
  #      'minibatch_size': 64,
  #      'random_seed': 6553,
  #      'tau': 0.005,
  #      'model_path': train_dir+"model.chkp",
  #      'enable_test': False, 
  #      'test_episodes': 10,
  #      'test_episodes_len': 5000}
  # actor =  DDPG(env, args=args)

  # model_path = os.path.split(args['model_path'])[0]+'/'
  # linear_func_model_name = 'K.model'
  # model_path = model_path+linear_func_model_name+'.npy'

  shield = Shield(env, None, model_path="./models", force_learning=True)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps,degree=6,aggressive=True)
  #shield.test_shield(10, 5000)

  #actor.sess.close()

if __name__ == "__main__":
  # learning_eposides = int(sys.argv[1])
  # actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  # critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  # train_dir = sys.argv[4]

  # K = [[ 0.08881306 -1.98026516]] This invokes two great shiels!
  oscillator("random_search", 500, 100, 0, [300, 200], [300, 250, 200], "ddpg_chkp/oscillator/300200300250200")