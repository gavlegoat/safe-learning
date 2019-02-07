from main import *

def robot (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([[0, 2],
    [1, 0]
    ])

  B = np.matrix([[1],
    [1]
    ])

  d, p = B.shape

  eq_err = 1e-2


  # the robot must move to a target region
  target_min = np.array([[ 8], [ 8]])
  target_max = np.array([[10], [10]])

  target = np.array([[0],[0]])
  for i in range(d):
    target[i, 0] = (target_min[i, 0] + target_max[i,0]) / 2
  print "target:\n {}".format(target)

  #intial state space
  s_min = np.array([[0.5],[0.5]])
  s_max = np.array([[1.5],[1.5]])

  S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  u_min = np.array([[-2.]])
  u_max = np.array([[ 2.]])

  # Unsafe regions!
  unsafe_1_min = np.array([[None], [None]])
  unsafe_1_max = np.array([[-2], [None]])
  avoid_1 = (unsafe_1_min, unsafe_1_max)

  unsafe_2_min = np.array([[10], [None]])
  unsafe_2_max = np.array([[None], [None]])
  avoid_2 = (unsafe_2_min, unsafe_2_max)

  unsafe_3_min = np.array([[None], [None]])
  unsafe_3_max = np.array([[None], [-2]])
  avoid_3 = (unsafe_3_min, unsafe_3_max)

  unsafe_4_min = np.array([[None], [10]])
  unsafe_4_max = np.array([[None], [None]])
  avoid_4 = (unsafe_4_min, unsafe_4_max)

  unsafe_5_min = np.array([[4], [4]])
  unsafe_5_max = np.array([[8], [8]])
  avoid_5 = (unsafe_5_min, unsafe_5_max)

  avoid_list = [avoid_1, avoid_2, avoid_3, avoid_4, avoid_5]

  # Evaluate whether a state is within the unsafe region
  def unsafef_eval(x):
    unsafe = True
    for (unsafe_x_min, unsafe_x_max) in avoid_list:
      unsafe = True
      for i in range(d):
        if unsafe_x_min[i, 0] is None and unsafe_x_max[i, 0] is None:
          continue 
        elif unsafe_x_min[i, 0] is not None and unsafe_x_max[i, 0] is None:
          if not (unsafe_x_min[i, 0] <= x[i, 0]):
            unsafe = False
            break
        elif unsafe_x_min[i, 0] is None and unsafe_x_max[i, 0] is not None:
          if not (x[i, 0] <= unsafe_x_max[i, 0]):
            unsafe = False
            break
        else:
          if not (unsafe_x_min[i, 0] <= x[i, 0] and x[i, 0] <= unsafe_x_max[i, 0]):
            unsafe = False
            break
      if unsafe:
        break
    return unsafe

  # Modify this to train a policy
  Q = np.matrix("0.0005 0 ; 0 0.0005")
  R = np.matrix("0")

  # Reward function
  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot((x-target).T,Q.dot(x-target))-np.dot(u.T,R.dot(u))
    # do not allow a robot to crash into an obstacle
    if (unsafef_eval(x)):
      reward -= 100
    return reward

  # Just for Testing
  def testf(x, u):
    if (unsafef_eval(x)):
      print "unsafe : {}".format(x)
      return -1
    return 0  

  # Just for Testing
  def test (Acl, coffset, K, x0, simulation_steps):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")
    reward = 0
    for t in time:
        reward += testf(xk, 0)
        xk =  Acl.dot(xk)+coffset
    #print "Score of the trace: {}".format(reward) 
    return reward

  # Just for Testing
  def random_test(K, simulation_steps):
    total_fails = 0
    succx = []
    succy = []
    failx = []
    faily = []

    Ks = np.split(K,[d],axis=1) 
    K = Ks[0]
    bias = Ks[1]

    Acl = A + B.dot(K)
    coffset = np.dot(B, bias)

    print "Acl = {}".format(Acl)
    print "coffset = {}".format(coffset)

    for i in range(100):
      x0 = np.matrix([
      [random.uniform(s_min[0, 0], s_max[0, 0])], 
      [random.uniform(s_min[1, 0], s_max[1, 0])]
      ])
      
      reward = test (Acl, coffset, K, x0, simulation_steps)

      if reward < 0:
        print "Failed on {}".format(x0)
        total_fails += 1
        failx.append(x0[0,0])
        faily.append(x0[1,0])
      else:
        succx.append(x0[0,0])
        succy.append(x0[1,0])
    print ("Among {} tests {} are failed.".format(100, total_fails))
    plt.plot(failx, faily, 'r')
    plt.plot(succx, succy, 'b')
    plt.show()


  # Learning Code
  while True:
    if K is None:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, rewardf=rewardf, bias=True)
    names = {0:"x0", 1:"x1"}
    draw_controller (A, B, K, x0, simulation_steps, names, rewardf=testf, bias=True)

    random_test (K, simulation_steps)

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     K = None
  #     continue

  #   Ks = np.split(K,[d],axis=1) 
  #   K = Ks[0]
  #   bias = Ks[1]
  #   print "K = {}".format(K)
  #   print "bias = {}".format(bias)

  #   Acl = A + B.dot(K)
  #   coffset = np.dot(B, bias)

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
  #     if unsafe_5_min[i, 0] is not None:
  #       unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(unsafe_5_min[i,0]) + ")*(" + str(unsafe_5_max[i,0]) + "-x[" + str(i+1) + "])")    
  #   for i in range(d):    
  #     if unsafe_5_min[i, 0] is not None:
  #       unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
  #   for i in range(d):
  #     if unsafe_5_min[i, 0] is not None:
  #       unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

    
  #   # Now we have init, unsafe and sysdynamics for verification
  #   sos = genSOS(d, ",".join(dxdt(Acl, coffset=coffset)), "\n".join(init), "\n".join(unsafe), 
  #                 "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr),degree=4)
  #   verified = verifySOS(writeSOS("1-robot.jl", sos), False, 900)
  #   print "verification result: {}".format(verified)
    break

#K = np.array([[-1.64062183,  0.98841361,  2.82362144]])
robot("random_search", 500, 50)