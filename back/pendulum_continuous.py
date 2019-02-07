from main import *

from shield import Shield
from Environment import Environment

import scipy.linalg as la
from pympc.dynamics.discrete_time_systems import mcais

# Show that there is an invariant that can prove the policy safe
def pendulum (learning_method, number_of_rollouts, simulation_steps, K=None):
  m = 1.
  l = 1.
  g = 10.

  #Dynamics that are continuous!
  A = np.matrix([
    [ 0., 1.],
    [g/l, 0.]
    ])
  B = np.matrix([
    [          0.],
    [1./(m*l**2.)]
    ])

  h = .01

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[-0.35],[-0.35]])
  s_max = np.array([[ 0.35],[ 0.35]])
  S0 = Polyhedron.from_bounds(s_min, s_max)

  #reward function
  Q = np.matrix([[1., 0.],[0., 1.]])
  R = np.matrix([[.005]])

  #safety constraint
  x_min = np.array([[-0.4],[-0.4]])
  x_max = np.array([[ 0.4],[ 0.4]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=True)
  shield = Shield(env, None, model_path="./models", force_learning=True)
  shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 1.0, step_size = 1.0, discretization=False)


  # names = {0:"pendulum position, meters", 1:"pendulum angle, radians"}

  # le = (K is None)

  # #sample an initial condition for system
  # z0 = random.uniform(s_min[0, 0], s_max[0, 0]) # initial position
  # v0 = random.uniform(s_min[1, 0], s_max[1, 0]) # initial velocity
  # #initial condition for system
  # #x0 = np.matrix([[.0],[.017]])
  # x0 = np.matrix([[z0], [v0]])
  # print ("Sampled initial state is:\n {}".format(x0))
  # #Interactive Verification
  # r = ""

  # while True:
  #   if le:
  #     K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, True, h, None, 1.0, 1.0)

  #   draw_controller (A, B, K, x0, simulation_steps, names, True, h)

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
  #     dimensions=[0,1]
  #     X.plot(dimensions, label=r'$D$', facecolor='b')
  #     O_inf.plot(dimensions, label=r'$\mathcal{O}_{\infty}$', facecolor='r')
  #     plt.legend()
  #     plt.show()
  #     ce = S0.is_included_in_with_ce(O_inf)
  #     if ce is None:
  #       print "Control Policy Verified!"
  #     else:
  #       print "A counter example is found {}".format(ce)
  #   else:
  #     #Generate the initial and unsafe conditions for verification
  #     d, p = Acl.shape

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
  #       verified = verifySOS(writeSOS("pendulum.jl", sos), False, 900)
  #       print "verification result: {}".format(verified)
  #   break

    # #Initial state condition
    # inits = ""
    # for i in range(d):
    #   inits = inits + ("-x[" + str(i+1) + "]^2")  
    # r = raw_input('Select a range to bound input space:')
    # #Iterative every unsafe condition
    # while True:
    #   verified = True
    #   for i in range(d):
    #     unsafe = "unsafe = x[" + str(i+1) + "]^2 - " + str(pow(max(abs(x_min[i,0]), abs(x_max[i,0])), 2))
    #     #Interactive Verification guided by user's satisfaction: 
    #     init = "init = " + str(pow(float(r), 2)) + inits
    #     # Now we have init, unsafe and sysdynamics for verification
    #     sos = genSOS(2, ",".join(dxdt(Acl)), init, unsafe)
    #     verified = verifySOS(writeSOS("pendulum.jl", sos), False, 180)
    #     if (verified is False):
    #       break
    #   if (verified is False):
    #     r = raw_input('Please Select a new range to bound input space:')
    #   else:
    #     break

    # r = raw_input('Satisfied with the result? (Y or N):')
    # if (r is "Y"):
    #   break
  #End of Interactive Verification

K = np.matrix([[-21.27932316, -19.03175129]])
pendulum ("random_search", 100, 200, K=K)