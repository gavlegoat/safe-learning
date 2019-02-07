from main import *

def car_platoon (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([
    [1,0,0,0,0,0,0],
    [0,1,0.1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0.1,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0.1],
    [0,0,0,0,0,0,1]
  ])
  B = np.matrix([
    [0.1,0,0,0],
    [0,0,0,0],
    [0.1,-0.1,0,0],
    [0,0,0,0],
    [0,0.1,-0.1,0],
    [0,0,0,0],
    [0,0,0.1,-0.1]
  ])

  eq_err = 1e-2

  #intial state space
  s_min = np.array([[ 19.9],[ 0.9], [-0.1], [ 0.9],[-0.1], [ 0.9], [-0.1]])
  s_max = np.array([[ 20.1],[ 1.1], [ 0.1], [ 1.1],[ 0.1], [ 1.1], [ 0.1]])

  #sample an initial condition for system
  x0 = np.matrix([
                    [random.uniform(s_min[0, 0], s_max[0, 0])], 
                    [random.uniform(s_min[1, 0], s_max[1, 0])],
                    [random.uniform(s_min[2, 0], s_max[2, 0])],
                    [random.uniform(s_min[3, 0], s_max[3, 0])], 
                    [random.uniform(s_min[4, 0], s_max[4, 0])],
                    [random.uniform(s_min[5, 0], s_max[5, 0])],
                    [random.uniform(s_min[6, 0], s_max[6, 0])]
                  ])
  print ("Sampled initial state is:\n {}".format(x0))  

  Q = np.matrix("1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 1 0; 0 0 0 0 0 0 1")
  R = np.matrix(".0005 0 0 0; 0 .0005 0 0; 0 0 .0005 0; 0 0 0 .0005")

  x_min = np.array([[18],[0.5],[-1],[0.5],[-1],[0.5],[-1]])
  x_max = np.array([[22],[1.5], [1],[1.5],[ 1],[1.5],[ 1]])
  u_min = np.array([[-10.], [-10.], [-10.], [-10.]])
  u_max = np.array([[ 10.], [ 10.], [ 10.], [ 10.]])

  #Coordination transformation
  origin = np.array([[20], [1], [0], [1], [0], [1], [0]])
  s_min -= origin
  s_max -= origin
  x0 -= origin
  x_min -= origin
  x_max -= origin

  #Coordination-transformed state space
  S0 = Polyhedron.from_bounds(s_min, s_max)

  if K is None:
    while True:
      K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max)
      names = {0:"x0", 1:"x1", 2:"x2", 3:"x3", 4:"x4", 5:"x5", 6:"x6"}
      draw_controller (A, B, K, x0, simulation_steps, names,coordination=origin)
      names = {0:"x0", 1:"x1"}
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
    names = {0:"x0", 1:"x1", 2:"x2", 3:"x3", 4:"x4", 5:"x5", 6:"x6"}
    draw_controller (A, B, K, x0, simulation_steps, names)
    names = {0:"x0", 1:"x1"}
    O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
    ce = S0.is_included_in_with_ce(O_inf)
    if ce is None:
      print "Control Policy Verified!"
    else:
      print "A counter example is found {}".format(ce)

car_platoon("random_search", 100, 200)