from main import *

from z3verify import *

def pendulum (learning_method, number_of_rollouts, simulation_steps, verification_steps, K=None):
  A = np.matrix([[1.9027,-1],
    [1,0]
    ])

  B = np.matrix([[1],
    [0]
    ])

  eq_err = 0

  #intial state space
  s_min = np.array([[-1.0],[-1.0]])
  s_max = np.array([[ 1.0],[ 1.0]])
  S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix(".0005")

  x_min = np.array([[-1.5],[-1.5]])
  x_max = np.array([[ 1.5],[ 1.5]])
  u_min = np.array([[-10.]])
  u_max = np.array([[ 10.]])

  result = synthesize_verifed_controller(x0, A, B, Q, R, 
                      eq_err, learning_method, 
                      number_of_rollouts, simulation_steps, verification_steps,
                      s_min, s_max, x_min, x_max, 
                      avoid_list=None, avoid_list_dynamic=None,
                      continuous=False, timestep=.01, rewardf=None, 
                      explore_mag=.04, step_size=.05, coffset=None, 
                      K=None)
  if result:
    print "A verified policy is learned!"
  else:
    print "Failed to find a verified policy."

#K = [[-0.01371043  0.00279629]]
pendulum("random_search", 100, 50, 10)