from main import *

def cooling (learning_method, number_of_rollouts, simulation_steps, K=None):
    A = np.matrix([
      [1.01,0.01,0],
      [0.01,1.01,0.01],
      [0.0,0.01,1.01]])
    B = np.matrix([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]])

    # amount of Gaussian noise in dynamics
    eq_err = 1e-2

     #intial state space
    s_min = np.array([[  16],[ 16], [16]])
    s_max = np.array([[  32],[ 32], [32]])
    S0 = Polyhedron.from_bounds(s_min, s_max)

    #sample an initial condition for system
    x0 = np.matrix([
        [random.uniform(s_min[0, 0], s_max[0, 0])], 
        [random.uniform(s_min[1, 0], s_max[1, 0])],
        [random.uniform(s_min[2, 0], s_max[2, 0])]
    ])
    print ("Sampled initial state is:\n {}".format(x0))

    Q = np.eye(3)
    R = np.eye(3)

    x_min = np.array([[-32],[-32],[-32]])
    x_max = np.array([[32],[32],[32]])
    u_min = np.array([[-15.],[-15.],[-15.]])
    u_max = np.array([[ 15.],[ 15.],[ 15.]])

    if K is None:
        while True:
          K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, explore_mag=0.02, step_size=0.0025)
          names = {0:"heat-1", 1:"heat-2", 2:"heat-3"}
          xk = draw_controller (A, B, K, x0, simulation_steps, names)
          names = {0:"heat-1", 2:"heat-2"}
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
        names = {0:"heat-1", 1:"heat-2", 2:"heat-3"}
        draw_controller (A, B, K, x0, simulation_steps, names)
        names = {0:"heat-1", 2:"heat-2"}
        O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
        ce = S0.is_included_in_with_ce(O_inf)
        if ce is None:
          print "Control Policy Verified!"
        else:
          print "A counter example is found {}".format(ce)

cooling("random_search", 50, 100)