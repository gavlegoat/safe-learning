from main import *

import math

# Show that there is an invariant that can prove the policy safe
def moutaincar (learning_method, number_of_rollouts, simulation_steps, K=None):

  #Dynamics that are defined as a continuous function!
  def f (x, u):
    delta = np.zeros((3, 1), float)
    #if (x[1, 0] < 0.6):
    delta[0, 0] = u[0, 0]*0.001 + math.cos(3*x[1,0])*(-0.0025)  #velocity
    delta[1, 0] = x[0, 0]                                       #position
    delta[2, 0] = 1                                             #time
    return delta

  #Closed loop system dynamics to text
  def f_to_str(K):
    kstr = K_to_str(K)
    f = []
    #dynamics for velocity
    f.append("(" + kstr[0] + ")*0.001 + (1 - 0.5*(3*x[2])^2) *(-0.0025)")
    #dynamics for position
    f.append("x[1]")
    #dynamics for time
    f.append("1")
    return f

  #User given safety property: after 2.5 seconds, the car position should reach higher than 0.59
  def unsafe_string():
    return ["0.59 - x[2]", "x[3] - 25"]

  #User given state space bound constraint 
  def  bound_string():
    return ["15 - x[2]", "x[2] - -15", "1 - x[2]", "x[2] - -1", "25 - x[3]", "x[3] - 0"]

  h = .1

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ 0.0],[-0.5], [0.0]])
  s_max = np.array([[ 0.0],[-0.5], [0.0]])
  S0 = Polyhedron.from_bounds(s_min, s_max)
  #sample an initial condition for system
  x0 = np.matrix([
    [random.uniform(s_min[0, 0], s_max[0, 0])], 
    [random.uniform(s_min[1, 0], s_max[1, 0])],
    [random.uniform(s_min[2, 0], s_max[2, 0])]
  ])
  print ("Sampled initial state is:\n {}".format(x0)) 

  #reward function
  Q = None
  R = np.zeros((1,1), float)
  np.fill_diagonal(R, 1)

  def rewardf(x, Q, u, R):
    reward = 0.0
    #Want to minimize the car driving force
    reward -= np.dot(u.T,R.dot(u))
    #User given safety property: after 2.5 seconds, the car position should reach higher than 0.59
    if x[1,0] < 0.6:
      reward -= 100#random.uniform(0.0,1.0)
    return reward

  names = {0:"car velocity", 1:"car positiom"}

  le = (K is None)

  while True:
    if le:
      K = random_search_helper (f, 3, 1, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, explore_mag = 0.4, step_size = 0.5)

    print "K = {}".format(K)

    draw_controller_helper (f, K, x0, 200, names, continuous=True, timestep=h)

    r = raw_input('Satisfied with the controller? (Y or N):')
    if (r is "N"):
      continue

    #Users are reuired to write a SOS program for verification
    #Specs for initial condions
    init = []
    initSOSPoly = []
    init_cnstr = []
    for i in range(3):
      init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")  
    for i in range(3):    
      initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
    for i in range(3):
      init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))

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
    bounds = bound_string()
    bound = []
    boundSOSPoly = []
    bound_cnstr = []
    for i in range(len(bounds)):
      bound.append("bound" + str(i+1) + " = " + bounds[i])
    for i in range(len(bounds)):
      boundSOSPoly.append("@variable m Zbound" + str(i+1) + " SOSPoly(Z)")
    for i in range(len(bounds)):
      bound_cnstr.append(" - Zbound" + str(i+1) + "*bound" + str(i+1))


    # Now we have init, unsafe and sysdynamics for verification
    sos = genSOSwithBound(3, ",".join(f_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(bound),
                  "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "\n".join(boundSOSPoly),
                  "".join(init_cnstr), "".join(unsafe_cnstr), "".join(bound_cnstr), degree=8)
    verified = verifySOS(writeSOS("moutaincar.jl", sos), False, 900)
    print "verification result: {}".format(verified)
    break

K = np.array([[-2.91581948, -7.30771812,  0.63870467]])
moutaincar ("random_search", 100, 1000, K)