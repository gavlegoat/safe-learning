from main import *

from z3verify import verify_controller_z3

import math

# Show that there is an invariant that can prove the policy safe
def collisionavoidance (learning_method, number_of_rollouts, simulation_steps, K=None):
  # two aeroplane crash avoidance system; 8-dimension and 2-input system
  ds = 8
  us = 2 

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    #We have two aeroplanes with 2 inputs for each controlling its own angular velocity!
    delta = np.zeros((ds, 1), float)
    delta[0, 0] = 0                         #velocity (always unchanged)
    delta[1, 0] = u[0, 0]                   #angular velocity (controlled by aeroplanes)
    delta[2, 0] = x[0, 0]*math.cos(x[1, 0]) #x-position
    delta[3, 0] = x[0, 0]*math.sin(x[1, 0]) #y-position

    delta[4, 0] = 0                         #velocity (always unchanged)
    delta[5, 0] = u[1, 0]                   #angular velocity (controlled by aeroplanes)
    delta[6, 0] = x[4, 0]*math.cos(x[5, 0]) #x-position
    delta[7, 0] = x[4, 0]*math.sin(x[5, 0]) #y-position

    return delta

  def taylorf (x, u):
    #We have two aeroplanes with 2 inputs for each controlling its own angular velocity!
    delta = np.zeros((ds, 1), float)
    delta[0, 0] = 0                         #velocity (always unchanged)
    delta[1, 0] = u[0, 0]                   #angular velocity (controlled by aeroplanes)
    #delta[2, 0] = x[0, 0]*(1 - 0.5*(pow(x[1,0],2)) + ((pow(x[1,0],4))/24) - (pow(x[1,0],6)/720))
    #delta[3, 0] = x[0, 0]*(x[1,0] - ((pow(x[1,0],3))/6) + (pow(x[1,0],5)/120))
    delta[2, 0] = x[0, 0]*(1 - 0.5*(pow(x[1,0],2)) + ((pow(x[1,0],4))/24))
    delta[3, 0] = x[0, 0]*(x[1,0] - ((pow(x[1,0],3))/6))

    delta[4, 0] = 0                         #velocity (always unchanged)
    delta[5, 0] = u[1, 0]                   #angular velocity (controlled by aeroplanes)
    #delta[6, 0] = x[4, 0]*(1 - 0.5*(pow(x[5,0],2)) + ((pow(x[5,0],4))/24) - (pow(x[5,0],6)/720))
    #delta[7, 0] = x[4, 0]*(x[5,0] - ((pow(x[5,0],3))/6) + (pow(x[5,0],5)/120))
    delta[6, 0] = x[4, 0]*(1 - 0.5*(pow(x[5,0],2)) + ((pow(x[5,0],4))/24))
    delta[7, 0] = x[4, 0]*(x[5,0] - ((pow(x[5,0],3))/6))

    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    f.append("0")
    f.append(kstr[0])
    f.append("x[1]*(1 - 0.5*(x[2]^2) + ((x[2]^4)/24))")
    f.append("x[1]*(x[2] - ((x[2]^3)/6))")

    f.append("0")
    f.append(kstr[1])
    f.append("x[5]*(1 - 0.5*(x[6]^2) + ((x[6]^4)/24))")
    f.append("x[5]*(x[6] - ((x[6]^3)/6))")
    return f

  h = 0.5

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  pi = 3.1415926

  #intial state space
  s_min = np.array([[.1],[pi*0.125],[ 0],[ 0], [-.2], [pi*0.125], [4.9], [4.9]])
  s_max = np.array([[.2],[pi*0.375],[.1],[.1], [-.1], [pi*0.375], [5.0], [5.0]])

  #the only portion of the entire state space that our verification is interested.
  bound_x_min = np.array([[.1],[     0],[0],[0], [-.2], [     0], [0], [0]])
  bound_x_max = np.array([[.2],[pi*0.5],[5],[5], [-.1], [pi*0.5], [5], [5]])

  #sample an initial condition for system
  x0 = np.matrix([
    [random.uniform(s_min[0, 0], s_max[0, 0])], 
    [random.uniform(s_min[1, 0], s_max[1, 0])],
    [random.uniform(s_min[2, 0], s_max[2, 0])],
    [random.uniform(s_min[3, 0], s_max[3, 0])],
    [random.uniform(s_min[4, 0], s_max[4, 0])], 
    [random.uniform(s_min[5, 0], s_max[5, 0])],
    [random.uniform(s_min[6, 0], s_max[6, 0])],
    [random.uniform(s_min[7, 0], s_max[7, 0])]
  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  S0 = Polyhedron.from_bounds(s_min, s_max)

  #reward functions
  Q = None
  R = np.zeros((us,us), float)
  np.fill_diagonal(R, 100)

  #user defined unsafety condition
  def unsafe_eval(x):
    distance = .25 - (pow((x[2,0]-x[6,0]), 2) + pow((x[3,0]-x[7,0]), 2))
    if (distance >= 0):
      return True
    #The following is suggested uncommented when the control policy is trained.
    outbound1 = -(x[1,0]-0)*(pi/2-x[1,0])
    if (outbound1 >= 0):
      return True
    outbound2 = -(x[5,0]-0)*(pi/2-x[5,0])
    if (outbound2 >= 0):
      return True
    return False

  def unsafe_string():
    return ["-(x[2]-0)*(3.1415926/2-x[2])"]

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(u.T,R.dot(u))
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
        [random.uniform(s_min[1, 0], s_max[1, 0])],
        [random.uniform(s_min[2, 0], s_max[2, 0])],
        [random.uniform(s_min[3, 0], s_max[3, 0])],
        [random.uniform(s_min[4, 0], s_max[4, 0])], 
        [random.uniform(s_min[5, 0], s_max[5, 0])],
        [random.uniform(s_min[6, 0], s_max[6, 0])],
        [random.uniform(s_min[7, 0], s_max[7, 0])]
      ])
      reward = test_controller_helper (f, K, x0, simulation_steps, testf, continuous=True, timestep=h)
      if reward < 0:
        print "Failed on {}".format(x0)
        total_fails += 1
    print ("Among {} tests {} are failed.".format(100, total_fails))

  def verify(x, initial_size, Theta, K, ds, s_min, s_max):
    #Users are reuired to write a SOS program for verification
    #Specs for initial condions
    init = []
    initSOSPoly = []
    init_cnstr = []
    for i in range(ds):
      init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")    
    for i in range(ds):    
      initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
    for i in range(ds):
      init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))

    for i in range(ds):
      l = x[i,0] - initial_size[i]
      h = x[i,0] + initial_size[i]
      init.append("init" + str(ds+i+1) + " = (x[" + str(i+1) + "] - (" + str(l) + "))*((" + str(h) + ")-x[" + str(i+1) + "])")    
    for i in range(ds):    
      initSOSPoly.append("@variable m Zinit" + str(ds+i+1) + " SOSPoly(Z)")
    for i in range(ds):
      init_cnstr.append(" - Zinit" + str(ds+i+1) + "*init" + str(ds+i+1))


    #Specs for unsafe conditions
    unsafes = unsafe_string()
    unsafe = []
    unsafeSOSPoly = []
    unsafe_cnstr = []
    for i in range(len(unsafes)):
      unsafe.append("unsafe" + str(i+1) + " = " + unsafes[i])
    for i in range(len(unsafes)):
      unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
    for i in range(len(unsafes)):
      unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

    #Spces for bounded state space
    bound = []
    boundSOSPoly = []
    bound_cnstr = []
    for i in range(ds):
      bound.append("bound" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(bound_x_min[i,0]) + ")*(" + str(bound_x_max[i,0]) + "-x[" + str(i+1) + "])")    
    for i in range(ds):    
      boundSOSPoly.append("@variable m Zbound" + str(i+1) + " SOSPoly(Z)")
    for i in range(ds):
      bound_cnstr.append(" - Zbound" + str(i+1) + "*bound" + str(i+1))

    sos = genSOSwithBound(ds, ",".join(f_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(bound),
                  "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "\n".join(boundSOSPoly),
                  "".join(init_cnstr), "".join(unsafe_cnstr), "".join(bound_cnstr))

    verified = verifySOS(writeSOS("collision-avoidance.jl", sos), False, 900)
    print "verification result: {}".format(verified)
    if (verified.find("Optimal") >= 0):
      return True
    else:
      return False

  names = {0:"v1", 1:"w1", 2:"x1", 3:"y1", 4:"v2", 5:"w2", 6:"x2", 7:"y2"}


  #Iteratively search polcies that can cover all initial states
  def verification_oracle(x, initial_size, Theta, K):
    return verify(x, initial_size, Theta, K, ds, s_min, s_max)

  def learning_oracle(x):
    if K is not None:
      return K
    else:
      return random_search_helper (f, ds, us, Q, R, x, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.04, step_size=0.05)

  def draw_oracle(x, K):
    return draw_controller_helper (f, K, x, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  initial_size = np.array([(s_max[i,0] - s_min[i,0])/20 for i in range(ds)])
  Theta = (s_min, s_max)
  result = verify_controller_z3(x0, initial_size, Theta, verification_oracle, learning_oracle, draw_oracle)

  if result:
    print "A verified policy is learned!"
  else:
    print "Failed to find a verified policy."

  # le = (K is None)

  # while True:
  #   if le:
  #     K = random_search_helper (f, ds, us, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag=0.04, step_size=0.05)

  #   print "K = {}".format(K)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   xk = draw_controller_helper (f, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  #   #print ("Random testing result on actually f:")
  #   #random_test(f, K, simulation_steps, continuous=True, timestep=h)

  #   print ("Random testing result on approximate f:")
  #   random_test(taylorf, K, simulation_steps, continuous=True, timestep=h)

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     continue

  #   unsafes = unsafe_string()
  #   verification_oracle(ds, s_min, s_max, unsafes, K)
    
  #   break

#K = [[-21.27932316, -19.03175129]]
K = np.array([[ 0.47683055, -2.22746533,  0.73605058, -1.8282028,  -0.23607473, -0.01584942,
   0.54546718, -0.40789043],
 [ 1.62339771,  1.01659839,  0.22537999, -0.14097677,  0.43943362, -2.21698874,
  -1.36152047,  1.35338043]])
collisionavoidance ("random_search", 1000, 100)