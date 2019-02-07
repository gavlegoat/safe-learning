from main import *

from shield import Shield
from Environment import PolySysEnvironment

import math

# Show that there is an invariant that can prove the policy safe
def biology (learning_method, number_of_rollouts, simulation_steps, K=None):
  # 10-dimension and 1-input system and 1-disturbance system
  ds = 18
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
    delta[10,0] = 5*x[9,0] - 5*x[10,0]
    delta[11,0] = 5*x[10,0] - 5*x[11,0]
    delta[12,0] = 5*x[11,0] - 5*x[12,0]
    delta[13,0] = 5*x[12,0] - 5*x[13,0]
    delta[14,0] = 5*x[13,0] - 5*x[14,0]
    delta[15,0] = 5*x[14,0] - 5*x[15,0]
    delta[16,0] = 5*x[15,0] - 5*x[16,0]
    delta[17,0] = 5*x[16,0] - 5*x[17,0]
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
    f.append("5*x[10]-5*x[11]")
    f.append("5*x[11]-5*x[12]")
    f.append("5*x[12]-5*x[13]")
    f.append("5*x[13]-5*x[14]")
    f.append("5*x[14]-5*x[15]")
    f.append("5*x[15]-5*x[16]")
    f.append("5*x[16]-5*x[17]")
    f.append("5*x[17]-5*x[18]")
    return f

  h = 0.01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[0.2],[-0.1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
  s_max = np.array([[0.3],[ 0.1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

  # #sample an initial condition for system
  # x0 = np.matrix([
  #   [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #   [random.uniform(s_min[1, 0], s_max[1, 0])],
  #   [random.uniform(s_min[2, 0], s_max[2, 0])], 
  #   [random.uniform(s_min[3, 0], s_max[3, 0])],
  #   [random.uniform(s_min[4, 0], s_max[4, 0])], 
  #   [random.uniform(s_min[5, 0], s_max[5, 0])],
  #   [random.uniform(s_min[6, 0], s_max[6, 0])], 
  #   [random.uniform(s_min[7, 0], s_max[7, 0])],
  #   [random.uniform(s_min[8, 0], s_max[8, 0])], 
  #   [random.uniform(s_min[9, 0], s_max[9, 0])],
  # ])
  # print ("Sampled initial state is:\n {}".format(x0))  


  Q = np.zeros((ds,ds), float)
  R = np.zeros((us,us), float)
  np.fill_diagonal(Q, 1)
  np.fill_diagonal(R, 1)

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[17,0] >= 0.05):
      return True
    return False
  def unsafe_string():
    return ["x[18] - 0.05"]

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

  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, timestep=h)

  ############ Train and Test NN model ############

  #################### Shield #################
  shield = Shield(env, None, "./models")
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 0.4, step_size = 0.5)

  # def random_test(f, K, simulation_steps, continuous=True, timestep=h):
  #   total_fails = 0
  #   for i in range(100):
  #     x0 = np.matrix([
   #    [random.uniform(s_min[0, 0], s_max[0, 0])], 
   #    [random.uniform(s_min[1, 0], s_max[1, 0])],
   #    [random.uniform(s_min[2, 0], s_max[2, 0])], 
   #    [random.uniform(s_min[3, 0], s_max[3, 0])],
   #    [random.uniform(s_min[4, 0], s_max[4, 0])], 
   #    [random.uniform(s_min[5, 0], s_max[5, 0])],
   #    [random.uniform(s_min[6, 0], s_max[6, 0])], 
   #    [random.uniform(s_min[7, 0], s_max[7, 0])],
   #    [random.uniform(s_min[8, 0], s_max[8, 0])], 
   #    [random.uniform(s_min[9, 0], s_max[9, 0])],
  #     ])
  #     reward = test_controller_helper (f, K, x0, simulation_steps, testf, continuous=True, timestep=h)
  #     if reward < 0:
  #       print "Failed on {}".format(x0)
  #       total_fails += 1
  #   print ("Among {} tests {} are failed.".format(100, total_fails))

  # names = {0:"x1",1:"x2",2:"x3",3:"x4",4:"x5",5:"x6",6:"x7",7:"x8",8:"x9",9:"x10"}

  # le = (K is None)

  # while True:
  #   if le:
  #     K = random_search_helper (f, ds, us, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.4, step_size=0.5)

  #   print "K = {}".format(K)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   draw_controller_helper (f, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  #   #print ("Random testing result on actually f:")
  #   #random_test(f, K, simulation_steps, continuous=True, timestep=h)

  #   #print ("Random testing result on taylor-approximative f:")
  #   #random_test(f, K, simulation_steps, continuous=True, timestep=h)

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
  #   # bound = []
  #   # boundSOSPoly = []
  #   # bound_cnstr = []
  #   # for i in range(ds):
  #   #   if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):
  #   #     bound.append("bound" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(bound_x_min[i,0]) + ")*(" + str(bound_x_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   # for i in range(ds):
  #   #   if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):    
  #   #     boundSOSPoly.append("@variable m Zbound" + str(i+1) + " SOSPoly(Z)")
  #   # for i in range(ds):
  #   #   if (bound_x_min[i,0] is not None and bound_x_max[i,0] is not None):
  #   #     bound_cnstr.append(" - Zbound" + str(i+1) + "*bound" + str(i+1))

  #   sos = genSOS(ds, ",".join(f_to_str(K)), "\n".join(init), "\n".join(unsafe),
  #                 "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly),
  #                 "".join(init_cnstr), "".join(unsafe_cnstr))
  #   verified = verifySOS(writeSOS("biology.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
  #   break

biology ("random_search", 200, 200)