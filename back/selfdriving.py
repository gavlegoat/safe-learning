from main import *

from shield import Shield
from Environment import PolySysEnvironment

import math

# Show that there is an invariant that can prove the policy safe
def selfdrive (learning_method, number_of_rollouts, simulation_steps, K=None):
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

  def f (x, u):
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
  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, bound_x_min=bound_x_min, bound_x_max=bound_x_max, timestep=0.1)

  shield = Shield(env, None, model_path="./models", force_learning=False)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps)

  # le = (K is None)

  # while True:
  #   if le:
  #     K = random_search_helper (f, ds, us, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.04, step_size=0.05)

  #   print "K = {}".format(K)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   draw_controller_helper (f, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  #   #print ("Random testing result on actually f:")
  #   #random_test(f, K, simulation_steps, continuous=True, timestep=h)

  #   #print ("Random testing result on taylor-approximative f:")
  #   #random_test(taylorf, K, simulation_steps, continuous=True, timestep=h)

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     continue

  #   #Users are reuired to write a SOS program for verification
  #   #Specs for initial condions
  #   init = []
  #   initSOSPoly = []
  #   init_cnstr = []
  #   for i in range(ds):
  #     init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   for i in range(ds):    
  #     initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(ds):
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

  #   #Spces for bounded state space
  #   bound = []
  #   boundSOSPoly = []
  #   bound_cnstr = []
  #   for i in range(ds):
  #     if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):
  #       bound.append("bound" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(bound_x_min[i,0]) + ")*(" + str(bound_x_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   for i in range(ds):
  #     if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):    
  #       boundSOSPoly.append("@variable m Zbound" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(ds):
  #     if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):
  #       bound_cnstr.append(" - Zbound" + str(i+1) + "*bound" + str(i+1))

  #   sos = genSOSwithBound(ds, ",".join(f_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(bound),
  #                 "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "\n".join(boundSOSPoly),
  #                 "".join(init_cnstr), "".join(unsafe_cnstr), "".join(bound_cnstr))
  #   verified = verifySOS(writeSOS("selfdrive.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
  #   break

selfdrive ("random_search", 100, 100)