from main import *

def pendulum (learning_method, number_of_rollouts, simulation_steps, K=None): 
  A = np.matrix([[1.00050004,0.01000167],
      [0.10001667,1.00050004]])
  B = np.matrix([[5.00041668e-05],
      [1.00016668e-02]])

  # amount of Gaussian noise in dynamics
  eq_err = 1e-2

  #intial state space
  s_min = np.array([[-0.4],[-0.4]])
  s_max = np.array([[ 0.4],[ 0.4]])
  S0 = Polyhedron.from_bounds(s_min, s_max)

  x0 = np.matrix([
    [random.uniform(s_min[0, 0], s_max[0, 0])], 
    [random.uniform(s_min[1, 0], s_max[1, 0])]
  ])

  Q = np.matrix([[1., 0.],[0., 1.]])
  R = np.matrix([[.005]])

  x_min = np.array([[-.5],[-.5]])
  x_max = np.array([[ .5],[ .5]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])

  if K is None:
    while True:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, False, .1, None, 1.0, 1.0)
      names = {0:"pendulum position, meters", 1:"pendulum angle, radians"}
      draw_controller (A, B, K, x0, simulation_steps, names)
      O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
      ce = S0.is_included_in_with_ce(O_inf)
      if ce is None:
        print "A verified policy is learned!"
        break
      else:
        print "Is the learned policy working well on the sampled input?: {}".format(O_inf.contains(x0))
        print "An input that is not within the current invariant set:\n {}".format(ce)
        x0 = np.asmatrix(ce)
  else:
    names = {0:"pendulum position, meters", 1:"pendulum angle, radians"}
    draw_controller (A, B, K, x0, simulation_steps, names)
    O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
    ce = S0.is_included_in_with_ce(O_inf)
    if ce is None:
      print "Control Policy Verified!"
    else:
      print "A counter example is found {}".format(ce)

pendulum ("random_search", 100, 200)