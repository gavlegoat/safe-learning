from main import *

from shield import Shield
from Environment import PolySysEnvironment

def lanekeep (learning_method, number_of_rollouts, simulation_steps, K=None):
  v0 = 27.7
  cf = 133000
  cr = 98800
  M  = 1650
  b  = 1.59
  a  = 1.11
  Iz = 2315.3

  ds = 4
  us = 2

  disturbance_x_min = np.array([[0],[0],[-0.035],[0]])
  disturbance_x_max = np.array([[0],[0],[ 0.035],[0]])

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    rd = random.uniform(-0.6, 0.6)
    delta = np.zeros((ds, 1), float)
    delta[0,0] = 1*x[1,0] + v0*x[2,0] + random.uniform(disturbance_x_min[0], disturbance_x_max[0])                                                                        #lateral displacement
    delta[1,0] = (-1*(cf+cr)/(M*v0))*x[1,0] + ((b*cr-a*cf)/(M*v0)-v0)*x[3,0] + (cf/M)*u[0,0] + random.uniform(disturbance_x_min[1], disturbance_x_max[1])                 #lateral velocity
    delta[2,0] = x[3,0] + random.uniform(disturbance_x_min[2], disturbance_x_max[2])                                                                                      #error yaw angle
    delta[3,0] = ((b*cr-a*cf)/(Iz*v0))*x[1,0] + (-1*(a*a*cf + b*b*cr)/(Iz*v0))*x[3,0] + (a*cf/Iz)*u[1,0]  + random.uniform(disturbance_x_min[3], disturbance_x_max[3])    #yaw rate

    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("1*x[2] + 27.7*x[3] + d[1]")
    f.append("(-1*(133000+98800)/(1650*27.7))*x[2] + ((1.59*98800-1.11*133000)/(1650*27.7)-27.7)*x[4] + (133000/1650)*{} + d[2]".format(kstr[0]))
    f.append("x[4] + d[3]")
    f.append("((1.59*98800-1.11*133000)/(2315.3*27.7))*x[2] + (-1*(1.11*1.11*133000 + 1.59*1.59*98800)/(2315.3*27.7))*x[4] + (1.11*133000/2315.3)*{} + d[4]".format(kstr[1]))
    return f

  h = 0.01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ -0.1],[ -0.1], [-0.1], [ -0.1]])
  s_max = np.array([[  0.1],[  0.1], [ 0.1], [  0.1]])
  # S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  # x0 = np.matrix([
  #     [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #     [random.uniform(s_min[1, 0], s_max[1, 0])],
  #     [random.uniform(s_min[2, 0], s_max[2, 0])],
  #     [random.uniform(s_min[3, 0], s_max[3, 0])]
  #   ])
  # print ("Sampled initial state is:\n {}".format(x0))

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005 0; 0 .0005")

  #user defined unsafety condition
  def unsafe_eval(x):
    if (x[0,0] > 0.9 or x[0, 0] < -0.9): # keep a safe distance from the car in front of you
      return True
    return False

  def unsafe_string():
    return ["-(x[1]- -0.9)*(0.9-x[1])"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))

    if (unsafe_eval(x)):
      reward -= 1
    return reward

  def testf(x, u):
    if (unsafe_eval(x)):
      return -1
    return 0 

  # Use sheild to directly learn a linear controller
  env = PolySysEnvironment(f, f_to_str,rewardf, testf, unsafe_string, ds, us, Q, R, s_min, s_max, disturbance_x_min=disturbance_x_min, disturbance_x_max=disturbance_x_max, timestep=h)

  shield = Shield(env, None, model_path="./models", force_learning=False)
  shield.train_polysys_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag=0.5, step_size=0.5, aggressive=True)

  # def random_test(f, K, simulation_steps, continuous=True, timestep=h):
  #   total_fails = 0
  #   for i in range(100):
  #     x0 = np.matrix([
  #     [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #     [random.uniform(s_min[1, 0], s_max[1, 0])],
  #     [random.uniform(s_min[2, 0], s_max[2, 0])],
  #     [random.uniform(s_min[3, 0], s_max[3, 0])]
  #     ])
  #     reward = test_controller_helper (f, K, x0, simulation_steps, testf, continuous=True, timestep=h)
  #     if reward < 0:
  #       print "Failed on {}".format(x0)
  #       total_fails += 1
  #   print ("Among {} tests {} are failed.".format(100, total_fails))

  # names = {0:"Displacement",1:"Velocity",2:"Error Yaw Angle",3:"Yaw Rate"}

  # le = (K is None)

  # while True:
  #   if le:
  #     K = random_search_helper (f, ds, us, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.4, step_size=0.5)

  #   print "K = {}".format(K)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   draw_controller_helper (f, K, x0, simulation_steps*10, names, continuous=True, timestep=h, rewardf=testf)

  #   #Verification of lane-keeping!
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

  #   sos = genSOS(ds, ",".join(f_to_str(K)), "\n".join(init), "\n".join(unsafe), 
  #                   "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
  #   verified = verifySOS(writeSOS("lanekeeping.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
  #   if (verified.find("Optimal") >= 0):
  #     return True
  #   else:
  #     return False

  #   break

K = np.array([[-4.21528005, -0.55237926, -9.45587692, -0.50038062],
 [-2.04298819,  0.50994964,  3.29331539, -1.02047674]])

lanekeep("random_search", 200, 200)