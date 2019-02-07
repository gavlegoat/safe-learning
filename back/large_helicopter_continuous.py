import os.path
import scipy.linalg as la
from main import *
from pympc.dynamics.discrete_time_systems import mcais

def helicopter (learning_method, number_of_rollouts, simulation_steps, K=None):
  HERE = os.path.dirname(os.path.abspath(__file__))
  filename_A = HERE+"/aux-matrices/helicopter_A.txt"
  filename_B = HERE+"/aux-matrices/helicopter_B.txt"

  # First read the continuous system from the txt file.
  with open(filename_A, 'r') as f:
      A_conti = []
      for line in f: # read rest of lines
          A_conti.append([float(x) for x in line.rstrip(', \n').strip(' ').split(',')])
  with open(filename_B, 'r') as f:
      B_conti = []
      for line in f: # read rest of lines
          B_conti.append([float(x) for x in line.rstrip(', \n').strip(' ').split(',')])

  A = np.array(A_conti)
  B = np.array(B_conti)

  eq_err = 1e-2

  h = .01

  #intial state space
  #initial_size = 0.004
  #0.1 - initial_size/10, 0.1 + initial_size/10
  s_min = np.array([[ 0.0996],[0.0996], [0.0996],[0.0996],[0.0996], [0.0996],[0.0996],[0.0996], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0]])
  s_max = np.array([[ 0.1004],[0.1004], [0.1004],[0.1004],[0.1004], [0.1004],[0.1004],[0.1004], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0],[0], [0],[0]])
  S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  x0 = np.array([
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
                    [random.uniform(s_min[15, 0], s_max[15, 0])], 
                    [random.uniform(s_min[16, 0], s_max[16, 0])],
                    [random.uniform(s_min[17, 0], s_max[17, 0])],
                    [random.uniform(s_min[18, 0], s_max[18, 0])], 
                    [random.uniform(s_min[19, 0], s_max[19, 0])],
                    [random.uniform(s_min[20, 0], s_max[20, 0])],
                    [random.uniform(s_min[21, 0], s_max[21, 0])], 
                    [random.uniform(s_min[22, 0], s_max[22, 0])],
                    [random.uniform(s_min[23, 0], s_max[23, 0])],
                    [random.uniform(s_min[24, 0], s_max[24, 0])],
                    [random.uniform(s_min[25, 0], s_max[25, 0])],
                    [random.uniform(s_min[26, 0], s_max[26, 0])],
                    [random.uniform(s_min[27, 0], s_max[27, 0])]
                  ])
  print ("Sampled initial state is:\n {}".format(x0))

  Q = np.zeros((28, 28), float)
  np.fill_diagonal(Q, 10000)

  R = np.zeros((6,6), float)
  np.fill_diagonal(R, .0005)

  x_min = np.array([[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8],[-8]])
  x_max = np.array([[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8],[ 8],[ 8], [8], [8]])
  u_min = np.array([[-1.],[-1.],[-1.],[-1.],[-1.],[-1.]])
  u_max = np.array([[ 1.],[ 1.],[ 1.],[ 1.],[ 1.],[ 1.]])

  X = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
 
  #compute the LQR gain
  K = -1*np.array(np.dot(scipy.linalg.inv(R), (np.dot(B.T,X))))

  le = (K is None)

  names = {0:"x0", 1:"x1", 2:"x2", 3:"x3"}

  while True:
    if le:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, True, h, None, 0.000001, 0.000001, lqr_start=True)

    draw_controller (A, B, K, x0, simulation_steps, names, True, h)

    r = raw_input('Satisfied with the controller? (Y or N):')
    if (r is "N"):
      continue

    #Generate the closed loop system for verification
    Acl = A + B.dot(K)
    print "Learned Closed Loop System: {}".format(Acl)

    if (True):
      #discretize the system for efficient verification
      X = Polyhedron.from_bounds(x_min, x_max)
      O_inf = mcais(la.expm(Acl * h), X, verbose=False)
      ce = S0.is_included_in_with_ce(O_inf)
      if ce is None:
        print "Control Policy Verified!"
      else:
        print "A counter example is found {}".format(ce)
    else:
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


      #Users are reuired to write a SOS program for verification
      #Specs for unsafe condions
      for i in range(4):
        unsafe = []
        unsafeSOSPoly = []
        unsafe_cnstr = []
        mid = (x_min[i, 0] + x_max[i, 0]) / 2
        radium = x_max[i, 0] - mid

        unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(mid) + ")^2 - " + str(pow(radium, 2)))    
        unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
        unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

        # Now we have init, unsafe and sysdynamics for verification
        sos = genSOS(d, ",".join(dxdt(Acl)), "\n".join(init), "\n".join(unsafe), 
                    "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
        verified = verifySOS(writeSOS("helicopter.jl", sos), False, 900)
        print "verification result: {}".format(verified)
    break

K = [[-1.15403339e-05,  6.00707704e-06, -8.14756733e-06,  7.20056495e-06,
  -1.74332251e-06,  9.76319416e-06, -8.45188231e-06, -1.46926157e-05,
  -2.55649170e-06, -6.52001861e-06, -7.72407639e-06,  8.43337599e-07,
   6.82538753e-06,  9.36713198e-06, -1.25252265e-05,  5.52100224e-06,
  -5.59869834e-06,  3.83642389e-06, -3.37607610e-06,  1.90622994e-06,
  -8.83692163e-06, -6.91804167e-06, -2.50931770e-06, -4.35308619e-06,
  -1.24363501e-07,  5.88589472e-06, -4.89715811e-06,  4.06562871e-06],
 [ 6.21853683e-07,  5.66070150e-07,  5.49141941e-06, -3.33037986e-07,
  -2.28945824e-06,  3.20557830e-06,  4.81573531e-06, -3.02652563e-07,
   5.58706385e-06, -3.52591663e-06, -4.26579878e-06, -9.27813878e-06,
   3.57484557e-06,  4.21861531e-06,  5.66140297e-06, -4.22629631e-06,
  -3.43632559e-06, -6.26250798e-07, -1.06442348e-05,  2.17456696e-06,
  -1.86585697e-06,  9.94238978e-06,  3.07148494e-07,  3.76873639e-06,
   7.05408593e-06,  7.91965761e-07,  7.57105272e-06, -8.32880959e-06],
 [-9.63335862e-06, -5.26288331e-06, -1.03121726e-05, -3.58419506e-07,
   1.35406966e-06,  2.89626018e-07, -4.77348870e-06, -6.08817633e-07,
  -1.46639321e-06, -7.64544101e-06,  9.44141010e-08,  1.22873979e-06,
   1.82731284e-05,  6.54750866e-06, -1.29620686e-06, -5.43157539e-06,
   7.98214446e-06,  5.94848349e-06,  3.67645517e-06,  5.81614672e-06,
   4.14894365e-06, -9.47844047e-06, -1.27755773e-05,  5.73200767e-06,
   2.65211972e-07, -2.37353241e-06, -3.94032436e-06, -3.31741704e-06],
 [-6.68498152e-07, -1.73809701e-05, -3.56637365e-06, -1.97952535e-05,
   4.59059001e-06, -2.30707056e-06, 3.14705709e-06,  3.28301351e-06,
  -1.73204906e-06,  1.44742657e-05,  4.92568793e-06,  1.72216238e-05,
   1.62519599e-06, -7.88491279e-06, -1.23851738e-05,  1.33965248e-05,
  -8.62299389e-07,  1.34112355e-06,  4.02505959e-06, -6.03903801e-06,
  -7.48341694e-06,  1.73077659e-07,  3.88512743e-06,  1.03152651e-05,
  -1.26597093e-05,  3.65895110e-06,  1.72615427e-06, -1.32928244e-06],
 [ 2.65054683e-06,  1.67519241e-06, -5.07864309e-06, -1.71734548e-05,
  -1.05573179e-05, -4.83360766e-06, -3.67621156e-06, -3.48380682e-06,
  -1.99000348e-05,  1.09961266e-05, -1.78549680e-06,  1.74573951e-05,
  -1.26163646e-05, -3.04962582e-06,  8.26692434e-07, -6.95013006e-06,
   5.42400272e-06,  2.13964036e-06,  3.51282800e-06,  1.80444327e-05,
   6.79040125e-06,  8.16910597e-06, -1.89859129e-05,  6.46065699e-06,
   2.90674780e-07,  3.19144081e-06, -1.12847604e-05,  6.22889672e-06],
 [ 8.78268579e-06,  1.13093113e-05, -5.58818589e-06,  1.23438789e-05,
   2.04650789e-05, 5.36122372e-06, -8.38044845e-06, -1.96287426e-06,
  -8.31689981e-07, -4.02818078e-06, -4.24636211e-06,  1.48685567e-05,
  -9.04811043e-06, -1.29117655e-05, -2.80440127e-06, -3.43666048e-06,
   4.09846400e-06, -9.02558059e-06, -1.39202354e-05, -3.42544628e-06,
  -6.84104820e-06, -6.04404427e-06,  1.57092407e-05,  8.27745650e-06,
  -5.02272344e-06, -3.28642812e-06, -3.06938459e-06,  3.38692364e-06]]
#helicopter("random_search", 100, 50000, K)
helicopter("random_search", 0, 500)

