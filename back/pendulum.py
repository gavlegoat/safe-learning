from main import *

def pendulum (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([[1.9027,-1],
    [1,0]
    ])

  B = np.matrix([[1],
    [0]
    ])

  eq_err = 0

  # initial action space
  u_min = np.array([[-1.]])
  u_max = np.array([[1.]])

  # intial state space
  s_min = np.array([[-0.5],[-0.5]])
  s_max = np.array([[ 0.5],[0.5]])
  x_min = np.array([[-2.0], [-2.0]])
  x_max = np.array([[2.0], [2.0]])

  S0 = Polyhedron.from_bounds(s_min, s_max)

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  Q = np.matrix("1 0 ; 0 1")
  R = np.matrix("1")

  names = {0:"x0", 1:"x1"}

  #Iteratively search polcies that can cover all initial states
  def verification_oracle(x, initial_size, Theta, K):
    O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
    min = np.array([[x[i,0] - initial_size[i]] for i in range(len(x))])
    max = np.array([[x[i,0] + initial_size[i]] for i in range(len(x))])
   
    S = Polyhedron.from_bounds(min, max)
    S = S.intersection(S0)
    ce = S.is_included_in_with_ce(O_inf)
    return (ce is None)

  def learning_oracle(x):
    if K is not None:
      return K
    else:
      return learn_controller (A, B, Q, R, x, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max)

  def  testf(x, u):
    if ((x < x_max).all() and (x > x_min).all()) and ((u < u_max).all() and (u > u_min).all()):
      return 0
    else:
      return -1

  def draw_oracle(x, K):
    result = test_controller (A, B, K, x, simulation_steps, rewardf=testf)
    return result

  Theta = (s_min, s_max)
  result = verify_controller_z3(x0, Theta, verification_oracle, learning_oracle, draw_oracle)

  if result:
    print "Control Policy Verified!"
  else:
    print "Control Policy Cannot Be Verified!"

#K = np.array([[-1.27117736, 0.72165111]])
pendulum("random_search", 100, 50)