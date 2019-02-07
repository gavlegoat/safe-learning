from main import *

# Show that there is an invariant that can prove the policy safe
def mcsr (learning_method, number_of_rollouts, simulation_steps, K=None):
  #Dynamics that are continuous!
  A = np.matrix([
    [ -18.925, 0., 0.22572, 80.823],
    [ 0., -18.925, -80.823, 0.22572],
    [ -0.22572, 80.823, -76.569, .0],
    [ -80.823, -0.22572, .0, -76.569]
    ])
  B = np.matrix([
    [5.3423,0],[0,-5.3423],[0.019038,6.8169],[6.8169,-0.019038]
    ])
  C = np.matrix([
    [5.3423, 0, -0.019038, -6.8169],
    [0, -5.3423, -6.8169, 0.019038]
    ])

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[0.0000784],[-0.0000588],[-0.0002295],[-0.0003803]])
  s_max = np.array([[0.0000980],[-0.0000392],[-0.0001531],[-0.0003039]])

  #sample an initial condition for system
  x0 = np.matrix([
    [random.uniform(s_min[0, 0], s_max[0, 0])], 
    [random.uniform(s_min[1, 0], s_max[1, 0])],
    [random.uniform(s_min[2, 0], s_max[2, 0])],
    [random.uniform(s_min[3, 0], s_max[3, 0])]
  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  S0 = Polyhedron.from_bounds(s_min, s_max)

  Q = np.zeros((4, 4), float)
  R = np.zeros((2, 2), float)

  #user defined unsafety condition
  def unsafe_eval(x):
    y = np.dot(C, x)
    if (y[0,0] - 0.35)*(0.4 - y[0,0]) >= 0 and (y[1,0]-0.45)*(0.6-y[1,0]) >= 0:
      return 0
    else:
      return -1
  def unsafe_string():
    y = dxdt(C)
    y1 = y[0]
    y2 = y[1]
    return ["({}-0.35)*(0.4-({}))".format(y1, y1), "({}-0.45)*(0.6-({}))".format(y2, y2)]

  def rewardf(x, Q, u, R):
    reward = np.array([[0]])
    unsafe = False
    if (unsafe_eval(x) >= 0):
      unsafe = True

    if (unsafe):
      reward -= 100
    return reward

  def testf(x, u):
    unsafe = False
    if (unsafe_eval(x) >= 0):
      unsafe = True

    if (unsafe):
      print "unsafe : {}".format(x)
      return -1
    return 0  

  names = {0:"x", 1:"y", 2:"z", 3:"a"}

  le = (K is None)

  while True:
    if le:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf)

    print ("If the the following trjectory socre is 0, then safe!")
    xk = draw_controller (A, B, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

    r = raw_input('Satisfied with the controller? (Y or N):')
    if (r is "N"):
      continue
    
    #Generate the closed loop system for verification
    Acl = A + B.dot(K)
    print "Learned Closed Loop System: {}".format(Acl)

    #Generate the initial and unsafe conditions for verification
    d, p = Acl.shape

    #Users are reuired to write a SOS program for verification
    #Specs for initial condions
    init = []
    initSOSPoly = []
    init_cnstr = []
    for i in range(d):
      init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")    
    for i in range(d):    
      initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
    for i in range(d):
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

    
    # Now we have init, unsafe and sysdynamics for verification
    sos = genSOS(d, ",".join(dxdt(Acl)), "\n".join(init), "\n".join(unsafe), 
                  "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
    verified = verifySOS(writeSOS("mcs4.jl", sos), False, 900)
    print "verification result: {}".format(verified)
    break

#K = [[-21.27932316, -19.03175129]]
K = [[ 0.8243813,  -0.50213305,  0.01882345, -1.09418288],
 [-0.5596807,  -0.43508471, -0.8530028,   0.29390881]]
mcsr ("random_search", 200, 500, K)