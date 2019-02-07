from main import *

from shield import Shield
from Environment import Environment

def cart_pole (learning_method, number_of_rollouts, simulation_steps, K=None):
    l = .22 # rod length is 2l
    m = (2*l)*(.006**2)*(3.14/4)*7856 # rod 6 mm diameter, 44cm length, 7856 kg/m^3
    M = .4
    dt = .02 # 20 ms
    g = 9.8

    A = np.matrix([[1, dt, 0, 0],[0,1, -(3*m*g*dt)/(7*M+4*m),0],[0,0,1,dt],[0,0,(3*g*(m+M)*dt)/(l*(7*M+4*m)),1]])
    B = np.matrix([[0],[7*dt/(7*M+4*m)],[0],[-3*dt/(l*(7*M+4*m))]])

    # amount of Gaussian noise in dynamics
    eq_err = 0

     #intial state space
    s_min = np.array([[ -0.1],[ -0.1], [-0.05], [ -0.05]])
    s_max = np.array([[  0.1],[  0.1], [ 0.05], [  0.05]])
    S0 = Polyhedron.from_bounds(s_min, s_max)

    #sample an initial condition for system
    x0 = np.matrix([
        [random.uniform(s_min[0, 0], s_max[0, 0])], 
        [random.uniform(s_min[1, 0], s_max[1, 0])],
        [random.uniform(s_min[2, 0], s_max[2, 0])],
        [random.uniform(s_min[3, 0], s_max[3, 0])]
      ])
    print ("Sampled initial state is:\n {}".format(x0))

    Q = np.matrix("1 0 0 0; 0 1 0 0 ; 0 0 1 0; 0 0 0 1")
    R = np.matrix(".0005")

    x_min = np.array([[-0.3],[-0.5],[-0.3],[-0.5]])
    x_max = np.array([[ .3],[ .5],[.3],[.5]])
    u_min = np.array([[-15.]])
    u_max = np.array([[ 15.]])

    env = Environment(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, continuous=False)
    shield = Shield(env, None, model_path="./models", force_learning=True)
    shield.train_shield(learning_method, number_of_rollouts, simulation_steps, eq_err=eq_err, explore_mag = 1.0, step_size = 1.0)

    # if K is None:
    #     while True:
    #       K = learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, x_min, x_max, explore_mag = 1.0, step_size = 1.0)
    #       names = {0:"cart position, meters", 1:"cart velocity", 2:"pendulum angle, radians", 3:"pendulum angle velocity"}
    #       draw_controller (A, B, K, x0, simulation_steps*10, names)
    #       names = {0:"cart position, meters", 2:"pendulum angle, radians"}
    #       O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
    #       ce = S0.is_included_in_with_ce(O_inf)
    #       if ce is None:
    #         print "A verified policy is learned!"
    #         break
    #       else:
    #         print "Is the learned policy working well on the sampled input?: {}".format(O_inf.contains(x0))
    #         print "An input that is not within the current invariant set:\n {}".format(ce)
    #         x0 = np.asmatrix(ce)
    # else:
    #     names = {0:"cart position, meters", 1:"cart velocity", 2:"pendulum angle, radians", 3:"pendulum angle velocity"}
    #     draw_controller (A, B, K, x0, simulation_steps*10, names)
    #     names = {0:"cart position, meters", 2:"pendulum angle, radians"}
    #     O_inf = verify_controller (np.asarray(A), np.asarray(B), np.asarray(K), x_min, x_max, u_min, u_max, names.keys())
    #     ce = S0.is_included_in_with_ce(O_inf)
    #     if ce is None:
    #       print "Control Policy Verified!"
    #     else:
    #       print "A counter example is found {}".format(ce)

#K = [[ 0.80005942,  0.98309886, 16.39761909,  2.6082432 ]]
#K = [[ 0.22723833  1.0023756  12.34160324  2.25333872]]
#cart_pole ("random_search", 1000, 1000)
cart_pole("random_search", 100, 200)