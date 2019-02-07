from main import *

from shield import Shield
from Environment import Environment

# Show that there is an invariant that can prove the policy safe
def navigation (learning_method, number_of_rollouts, simulation_steps, K=None):
  #Dynamics that are continuous!
  A = np.matrix([
    [ 0., 0., 1., 0., 0., 0., 0., 0.],
    [ 0., 0., 0., 1., 0., 0., 0., 0.],
    [ 0., 0., -1.2, .1, 0., 0., 0., 0.],
    [ 0., 0., .1, -1.2, 0., 0., 0., 0.],
    [ 0., 0., 0., 0., 0., 0., 1., 0.],
    [ 0., 0., 0., 0., 0., 0., 0., 1.],
    [ 0., 0., 0., 0., 0., 0., -1.2, .1],
    [ 0., 0., 0., 0., 0., 0., .1, -1.2]
    ])
  B = np.matrix([
    [0,0,0,0],
    [0,0,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,1,0],
    [0,0,0,1]
    ])

  d, p = B.shape

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  target = np.array([[0],[5], [0], [0], [0], [6], [0], [0]])

  #intial state space
  s_min = np.array([[-2.1],[-4.1],[0],[0], [1.9], [-4.1], [0], [0]])
  s_max = np.array([[-1.9],[-3.9],[0],[0], [2.1], [-3.9], [0], [0]])
  s_min -= target
  s_max -= target

  coffset = np.dot(A, target)
  print "coffset:\n {}".format(coffset)

  #reward functions
  Q = np.zeros((d,d), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((p,p), float)
  np.fill_diagonal(R, .005)

  #user defined unsafety condition
  def unsafe_eval(x):
    return 0.25 - (pow((x[0]-x[4]+target[0,0]-target[4,0]), 2) + pow((x[1]-x[5]+target[1,0]-target[5,0]), 2))
  def unsafe_string():
    return ["0.25 - (x[1]-x[5])^2 - (x[2]-x[6]-1)^2"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    # Do not allow two cars crashing with each other
    distance = pow((x[0]-x[4]+target[0,0]-target[4,0]), 2) + pow((x[1]-x[5]+target[1,0]-target[5,0]), 2)
    if (unsafe_eval(x) >= 0):
      reward -= 900
    return reward

  def testf(x, u):
    unsafe = True
    if (unsafe_eval(x) < 0):
      unsafe = False
    if (unsafe):
      print "unsafe : {}".format(x+target)
      return -1
    return 0  

  u_min = np.array([[-50.], [-50]])
  u_max = np.array([[ 50.], [ 50]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, None, None, Q, R, 
            continuous=True, rewardf=rewardf, unsafe=True, unsafe_property=unsafe_string,
            terminal_err=0.1, bad_reward=-100)

  ############ Train and Test NN model ############

  #################### Shield #################
  shield = Shield(env, None, "./models")
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, rewardf=rewardf, testf=testf, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5, bias=False)
  #shield.test_shield(1000, 5000)

  #sample an initial condition for system
  # x0 = np.matrix([
  #   [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #   [random.uniform(s_min[1, 0], s_max[1, 0])],
  #   [random.uniform(s_min[2, 0], s_max[2, 0])],
  #   [random.uniform(s_min[3, 0], s_max[3, 0])],
  #   [random.uniform(s_min[4, 0], s_max[4, 0])], 
  #   [random.uniform(s_min[5, 0], s_max[5, 0])],
  #   [random.uniform(s_min[6, 0], s_max[6, 0])],
  #   [random.uniform(s_min[7, 0], s_max[7, 0])]
  # ])
  # print ("Sampled initial state is:\n {}".format(x0))  
  # x0 -= target

  #names = {0:"x1", 1:"y1", 2:"vx1", 3:"vy1", 4:"x2", 5:"y2", 6:"vx2", 7:"vy2"}

  # le = (K is None)

  # while True:
  #   if le:
  #     K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.4, step_size=0.5, coffset=coffset)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   xk = draw_controller (A, B, K, x0, simulation_steps*10, names, continuous=True, timestep=h, rewardf=testf, coordination=target, coffset=coffset)

  #   print ("Arrived at\n {}".format(xk+target))

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     continue
    
  #   #Generate the closed loop system for verification
  #   Acl = A + B.dot(K)

  #   d, p = Acl.shape

  #   #Users are reuired to write a SOS program for verification
  #   #Specs for initial condions
  #   init = []
  #   initSOSPoly = []
  #   init_cnstr = []
  #   for i in range(d):
  #     init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   for i in range(d):    
  #     initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(d):
  #     init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))

  #   #Specs for unsafe conditions
  #   unsafes = unsafe_string()
  #   unsafe = []
  #   unsafeSOSPoly = []
  #   unsafe_cnstr = []
  #   for i in range(len(unsafes)):
  #     unsafe.append("unsafe" + str(i+1) + " = " + unsafes[i])
  #   for i in range(len(unsafes)):
  #     unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(len(unsafes)):
  #     unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

  #   sos = genSOS(d, ",".join(dxdt(Acl, coffset=coffset)), "\n".join(init), "\n".join(unsafe), 
  #                 "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
  #   verified = verifySOS(writeSOS("2-cars-navigation.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
  #   break

#K = [[-21.27932316, -19.03175129]]
K = [[  0.30923238,  -8.21860525,  -3.55703702, -10.34199675,   2.44158251,
    2.89652896,  10.77504469,   5.88230113],
 [  8.6784715,   -7.6048211,    6.16931975, -11.17736349,  -3.08448797,
   -0.52737569,  -1.87705704,   3.7769782 ],
 [-15.82419075,   1.84858296,  -1.84047885,  -3.68344919,   1.04760547,
   10.4921827,   -8.3316463,   11.01894962],
 [ -4.43962048,   0.20532201,  -6.85347123,  -5.4735437,    2.8929692,
  -11.30919106,  -7.42490079,  -8.19019377]]
navigation ("random_search", 1000, 100, K)