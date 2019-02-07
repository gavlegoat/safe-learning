from main import *

from shield import Shield
from Environment import Environment

# Show that there is an invariant that can prove the policy safe
def navigation (learning_method, number_of_rollouts, simulation_steps, K=None):
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

  d, p = B.shape

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 0

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

  min_avoid_array = np.array([unsafe_x_min])
  max_avoid_array = np.array([unsafe_x_max])

  #reward functions
  Q = np.zeros((d, d), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((p,p), float)
  np.fill_diagonal(R, .005)

  u_min = np.array([[-50], [-50]])
  u_max = np.array([[ 50], [ 50]])

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    #penalize the distance toward the center of unsafe
    unsafe = True
    for i in range(d):
      if unsafe_x_min[i, 0] != np.NINF:
        if not (unsafe_x_min[i, 0] <= x[i, 0] and x[i, 0] <= unsafe_x_max[i, 0]):
          unsafe = False
          break
    if (unsafe):
      reward -= 100
    return reward

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

  env = Environment(A, B, u_min, u_max, s_min, s_max, min_avoid_array, max_avoid_array, Q, R, continuous=True, rewardf=rewardf, unsafe=True, terminal_err=0.1, bad_reward=-100)

  ############ Train and Test NN model ############

  #################### Shield #################
  shield = Shield(env, None, "./models")
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, testf=testf, eq_err=eq_err, explore_mag = 0.1, step_size = 0.1, bias=False, lqr_start=True)
  #shield.test_shield(1000, 5000)


  #sample an initial condition for system
  # x0 = np.matrix([
  #   [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #   [random.uniform(s_min[1, 0], s_max[1, 0])],
  #   [random.uniform(s_min[2, 0], s_max[2, 0])],
  #   [random.uniform(s_min[3, 0], s_max[3, 0])]
  # ])
  # print ("Sampled initial state is:\n {}".format(x0)) 
  # x0 -= target

  #names = {0:"x", 1:"y", 2:"vx", 3:"vy"}

  #draw_controller (A, B, K, x0, simulation_steps*10, names, continuous=True, timestep=h, rewardf=testf, coordination=target)

  # le = (K is None)

  # while True:
  #   if le:
  #     K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.4, step_size=0.5)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   xk = draw_controller (A, B, K, x0, simulation_steps*10, names, continuous=True, timestep=h, rewardf=testf, coordination=target)

  #   print ("Arrived at {}".format(xk+target))

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     continue
    
  #   #Generate the closed loop system for verification
  #   Acl = A + B.dot(K)
  #   print "Learned Closed Loop System: {}".format(Acl)

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


  #   #Users are reuired to write a SOS program for verification
  #   #Specs for unsafe condions
  #   unsafe = []
  #   unsafeSOSPoly = []
  #   unsafe_cnstr = []
  #   for i in range(d):
  #     if unsafe_x_min[i, 0] is not None:
  #       unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(unsafe_x_min[i,0]) + ")*(" + str(unsafe_x_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   for i in range(d):    
  #     if unsafe_x_min[i, 0] is not None:
  #       unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(d):
  #     if unsafe_x_min[i, 0] is not None:
  #       unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

    
  #   # Now we have init, unsafe and sysdynamics for verification
  #   sos = genSOS(d, ",".join(dxdt(Acl)), "\n".join(init), "\n".join(unsafe), 
  #                 "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
  #   verified = verifySOS(writeSOS("1-car-navigation.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
  #   break

#K = [[-21.27932316, -19.03175129]]
#Correct K
#K = np.array([[ -2.757318,     8.14072623,  -3.40473968,   3.35465468],
#    [ -5.38419094, -10.11003801,  -3.04579954,  -8.17257855]])
#Error K
# K = np.array([[ -5.8593484,   -3.44902618,  -6.41939682,  -5.95818385],
#  [ -0.98328122, -13.18940486,   7.53294555,  -8.48188398]])
#K = [[-9.30297954,  8.74752577, -9.02805432,  0.91674678],
#    [-5.83383704, -8.22630367, -3.51927504, -7.28304629]]
navigation ("random_search", 200, 200)