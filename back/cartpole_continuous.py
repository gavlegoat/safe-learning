
from main import *

from shield import Shield
from Environment import Environment

import scipy.linalg as la
from pympc.dynamics.discrete_time_systems import mcais

def cartpole (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([
  [0, 1,     0, 0],
  [0, 0, 0.716, 0],
  [0, 0,     0, 1],
  [0, 0, 15.76, 0]
  ])
  B = np.matrix([
  [0],
  [0.9755],
  [0],
  [1.46]
  ])

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 0

   #intial state space
  s_min = np.array([[ -0.05],[ -0.1], [-0.05], [ -0.05]])
  s_max = np.array([[  0.05],[  0.1], [ 0.05], [  0.05]])
  S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  # x0 = np.matrix([
  #     [random.uniform(s_min[0, 0], s_max[0, 0])], 
  #     [random.uniform(s_min[1, 0], s_max[1, 0])],
  #     [random.uniform(s_min[2, 0], s_max[2, 0])],
  #     [random.uniform(s_min[3, 0], s_max[3, 0])]
  #   ])
  # print ("Sampled initial state is:\n {}".format(x0))

  x_min = np.array([[-0.3],[-0.45],[-0.3],[-0.45]])
  x_max = np.array([[ .3],[ .45],[.3],[.45]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
  R = np.matrix(".0005")

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)
  shield = Shield(env, None, model_path="./models", force_learning=True)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 1.0, step_size = 1.0, discretization=False)

  # d, p = B.shape

  #def rewardf(x, Q, u, R):
  # reward = 0
  # reward += -np.dot(x.T,Q.dot(x))
  # return reward

  # def testf(x, u):
  #   unsafe = False
  #   for i in range(d):
  #     if not (x_min[i, 0] <= x[i, 0] and x[i, 0] <= x_max[i, 0]):
  #       unsafe = True
  #       break
  #   if (unsafe):
  #     return -1
  #   return 0

  # names = {0:"cart position, meters", 2:"pendulum angle, radians"}

  # le = (K is None)

  # while True:
  #   if le:
  #     K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, 
  #       continuous=True, timestep=h, rewardf=rewardf, explore_mag = 1.0, step_size = 1.0)

  #   print ("If the the following trjectory socre is 0, then safe!")
  #   draw_controller (A, B, K, x0, simulation_steps, names, continuous=True, timestep=h, rewardf=testf)

  #   r = raw_input('Satisfied with the controller? (Y or N):')
  #   if (r is "N"):
  #     continue
    
  #   #Generate the closed loop system for verification
  #   Acl = A + B.dot(K)
  #   print "Learned Closed Loop System: {}".format(Acl)

  #   if (False):
  #     #discretize the system for efficient verification
  #     X = Polyhedron.from_bounds(x_min, x_max)
  #     O_inf = mcais(la.expm(Acl * h), X, verbose=False)
  #     dimensions=[0,2]
  #     X.plot(dimensions, label=r'$D$', facecolor='b')
  #     O_inf.plot(dimensions, label=r'$\mathcal{O}_{\infty}$', facecolor='r')
  #     plt.legend()
  #     plt.show()
  #     ce = S0.is_included_in_with_ce(O_inf)
  #     if ce is None:
  #       print "Control Policy Verified!"
  #     else:
  #       print "A counter example is found {}".format(ce)
  #   elif True:
  #     #Users are reuired to write a SOS program for verification
  #     #Specs for initial condions
  #     init = []
  #     initSOSPoly = []
  #     init_cnstr = []
  #     for i in range(d):
  #       init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(s_min[i,0]) + ")*(" + str(s_max[i,0]) + "-x[" + str(i+1) + "])")    
  #     for i in range(d):    
  #       initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
  #     for i in range(d):
  #       init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))


  #     #Users are reuired to write a SOS program for verification
  #     #Specs for unsafe condions
  #     for i in range(d):
  #       unsafe = []
  #       unsafeSOSPoly = []
  #       unsafe_cnstr = []
  #       mid = (x_min[i, 0] + x_max[i, 0]) / 2
  #       radium = x_max[i, 0] - mid

  #       unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(mid) + ")^2 - " + str(pow(radium, 2)))    
  #       unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
  #       unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))


  #       # Now we have init, unsafe and sysdynamics for verification
  #       sos = genSOS(d, ",".join(dxdt(Acl)), "\n".join(init), "\n".join(unsafe), 
  #                   "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), "".join(unsafe_cnstr))
  #       verified = verifySOS(writeSOS("cartpole.jl", sos), False, 900)
  #       print "verification result: {}".format(verified)
  #   break

#K = np.array([[  5.40839089,  12.07876949, -61.3628737,  -18.08569268]])
cartpole("random_search", 200, 500)