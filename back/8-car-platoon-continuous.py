from main import *

def car_platoon (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,1,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,1,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,1,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,1, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,1, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,1, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,1],
  [0, 0,0,   0,0,   0,0,   0,0, 0,0, 0,0, 0,0]
  ])
  B = np.matrix([
  [1,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [1,  -1,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   1,  -1,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   1,  -1,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   1,  -1,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   1,  -1,   0,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   1,  -1,   0],
  [0,   0,   0,   0,   0,   0,   0,   0],
  [0,   0,   0,   0,   0,   0,   1,  -1],
  ])

  h = .1

  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ 19.9],[ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1], [ 0.9], [-0.1]])
  s_max = np.array([[ 20.1],[ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1], [ 1.1], [ 0.1]])

  x_min = np.array([[18],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1],[0.5],[-1]])
  x_max = np.array([[22],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5], [1],[1.5],[ 1],[1.5],[ 1],[1.5],[ 1]])

  target = np.array([[20],[1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                    [random.uniform(s_min[2, 0], s_max[2, 0])],
                    [random.uniform(s_min[3, 0], s_max[3, 0])], 
                    [random.uniform(s_min[4, 0], s_max[4, 0])],
                    [random.uniform(s_min[5, 0], s_max[5, 0])],
                    [random.uniform(s_min[6, 0], s_max[6, 0])],
                    [random.uniform(s_min[7, 0], s_max[7, 0])],
                    [random.uniform(s_min[8, 0], s_max[8, 0])],
                    [random.uniform(s_min[9, 0], s_max[9, 0])],
                    [random.uniform(s_min[10, 0], s_max[10, 0])],
                    [random.uniform(s_min[11, 0], s_max[11, 0])],
                    [random.uniform(s_min[12, 0], s_max[12, 0])],
                    [random.uniform(s_min[13, 0], s_max[13, 0])],
                    [random.uniform(s_min[14, 0], s_max[14, 0])],
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  s_min -= target
  s_max -= target
  x_min -= target
  x_max -= target
  x0 -= target

  #Coordination-transformed state space
  S0 = Polyhedron.from_bounds(s_min, s_max)

  coffset = np.dot(A, target)
  print "coffset:\n {}".format(coffset)

  Q = np.zeros((15, 15), float)
  np.fill_diagonal(Q, 1)

  R = np.zeros((8,8), float)
  np.fill_diagonal(R, 1)

  d, p = B.shape

  def rewardf(x, Q, u, R):
    reward = 0
    reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
    return reward

  def testf(x, u):
    unsafe = False
    for i in range(d):
      if not (x_min[i, 0] <= x[i, 0] and x[i, 0] <= x_max[i, 0]):
        unsafe = True
        break
    if (unsafe):
      return -1
    return 0

  names = {0:"x0", 1:"x1", 2:"x2", 3:"x3", 4:"x4", 5:"x5", 6:"x6", 7:"x7", 8:"x8", 9:"x9", 10:"x10", 11:"x11", 12:"x12", 13:"x13", 14:"x14"}

  le = (K is None)

  while True:
    if le:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, continuous=True, timestep=h, rewardf=rewardf, coffset=coffset)

    print ("If the the following trjectory socre is 0, then safe!")
    xk = draw_controller (A, B, K, x0, simulation_steps*10, names, continuous=True, timestep=h, rewardf=testf, coordination=target, coffset=coffset)

    print ("Arrived at {}".format(xk+target))

    r = raw_input('Satisfied with the controller? (Y or N):')
    if (r is "N"):
      continue
    
    #Generate the closed loop system for verification
    Acl = A + B.dot(K)
    print "Learned Closed Loop System: {}".format(Acl)

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


    #Users are reuired to write a SOS program for verification
    #Specs for unsafe condions
    for i in range(d):
      unsafe = []
      unsafeSOSPoly = []
      unsafe_cnstr = []
      mid = (x_min[i, 0] + x_max[i, 0]) / 2
      radium = x_max[i, 0] - mid

      unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(mid) + ")^2 - " + str(pow(radium, 2)))    
      unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
      unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

      # Now we have init, unsafe and sysdynamics for verification
      sos = genSOS(d, ",".join(dxdt(Acl, coffset=coffset)), "\n".join(init), "\n".join(unsafe), 
                  "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
      verified = verifySOS(writeSOS("8-car-platoon.jl", sos), False, 900)
      print "verification result: {}".format(verified)
    break

K = [[-4.19315062e-01, -5.39151248e-01, -8.32606314e-01,  1.79214710e-01,
  -2.22166970e-01,  7.66510925e-02,  4.31777139e-01, -6.64724023e-01,
  -7.28285825e-01, -2.25857782e-01, -1.02424602e-01, -1.75030476e-01,
  -1.07903207e+00, -3.35756739e-01,  3.35782723e-01],
 [-1.90683135e-01,  6.28939791e-01,  1.00857012e+00, -7.04281412e-01,
  -6.33951849e-01, -4.28401259e-02, -5.88829448e-01,  8.82968040e-01,
   4.19738086e-01,  6.01144448e-01, -1.04536861e+00,  6.55905364e-01,
   6.62965089e-01,  3.05366875e-01, -2.37897381e-02],
 [-4.19856913e-01, -1.17949103e-01,  5.43544330e-02,  8.80998556e-01,
   1.08115368e+00, -2.25946929e-01,  2.50488533e-01,  7.86394647e-02,
  -1.15355379e+00, -2.56608546e-01, -1.60517383e-01,  3.09887842e-02,
   6.94614262e-02, -1.04538718e+00, -1.07131347e-01],
 [ 6.20022053e-01,  3.59639972e-02,  5.17266758e-01,  9.44569927e-01,
  -4.96727049e-01,  6.55381895e-01,  1.46402527e+00, -1.45475394e+00,
   1.04137495e-01, -5.44735666e-01, -2.93727936e-02, -5.42880607e-01,
   2.26351593e-01, -1.02841597e-01, -1.95012500e-01],
 [ 2.39463282e-01, -6.94011777e-01,  6.29487479e-02,  5.06479068e-01,
   8.10619313e-01,  4.89113216e-01, -6.27421223e-02,  6.31501986e-01,
   9.59526911e-01, -4.77258559e-01, -1.00365453e+00, -3.53960987e-01,
   1.29313295e-01, -4.01298023e-02,  7.49458343e-01],
 [-6.61342548e-01,  4.45347287e-01,  4.60106879e-01,  3.51552440e-01,
  -4.00254928e-01, -1.48475418e-01,  8.68168819e-01, -7.29062003e-02,
   3.32749998e-01,  3.91309374e-01,  6.33506778e-01, -3.85041002e-01,
  -6.39433258e-01, -2.85171888e-01, -3.48341805e-01],
 [-6.95996810e-01, -6.31227721e-01, -6.95425526e-01, -2.13730410e-01,
   4.02403604e-01,  4.56750904e-01,  8.66145120e-01,  2.83131604e-01,
  -1.49169345e-01,  6.64578375e-01,  5.70778470e-02,  5.33811356e-01,
   7.43810610e-01, -7.83046745e-01, -3.47483359e-01],
 [ 2.09074899e-01, -1.39068240e-01, -1.07836334e+00, -1.08671367e-03,
  -6.76151342e-02,  6.00685391e-01,  2.63008776e-01, -7.13134351e-01,
  -5.61612041e-01,  1.47083389e-01, -1.44165667e-01,  2.05005373e-01,
   2.59131259e-01,  4.89704579e-01,  5.46350465e-01]]
car_platoon("random_search", 200, 500, K)