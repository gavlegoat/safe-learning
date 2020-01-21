import metrics
from metrics import timeit
from main import *
import scipy.optimize
import Environment

import os
import re

import synthesis

class Shield(object):
    """A safe controller for an environment.

    This class represents a disjunctive linear controller for some environment.
    That is, the controller consists of a list of pairs of polytopes and
    matrices. To apply the controller at a particular state, we choose a
    polytope that the state is in and multiply the state by the corresponding
    matrix.

    Note that a polytope is represented as a matrix A and a vector b where the
    polytope includes all points x such that A * x <= b.

    Attributes:
        K_list (list of matrices): The linear controllers for this shield.
        inv_list (list of polytopes): The spaces associated with each controller.
    """

    def __init__(self, env, K_list=None, inv_list=None, cover_list=None, bound=20):
        """Initialize a new Shield.

        If K_list and inv_list are given, they are used as the new shield.
        Otherwise, the new shield is initialized empty and must be trained
        with train_shield().

        Arguments:
            env (Environment): The environment under control.
            actor (ActorNetwork): The actor network.

        Keyword arguments:
            K_list (list of matrices): The initial controllers.
            inv_list (list of polytopes): The initial invariants.
        """
        self.env = env

        self.K_list = [] if K_list is None else K_list
        self.inv_list = [] if inv_list is None else inv_list
        self.cover_list = [] if cover_list is None else cover_list

        if K_list is not None:
            self.set_covers(bound)

        self.last_shield = -1

    def set_covers(self, bound=20):
        self.use_list = []
        dt = self.env.timestep if self.env.continuous else 0.01
        if isinstance(self.env, Environment.PolySysEnvironment):
            unsafe_space = []
            for (A, b) in zip(self.env.unsafe_A, self.env.unsafe_b):
                unsafe_space.append((A.tolist(), np.asarray(b).flatten().tolist()))
            env = (self.env.capsule, self.env.continuous, dt, unsafe_space)
        else:
            # unsafe_space format: [(matrix, vector}]
            unsafe_space = []
            safe_min = self.env.x_min
            safe_max = self.env.x_max
            for i in range(len(safe_min)):
                A = [[0.0] * len(safe_min)]
                A[0][i] = -1.0
                b = [-safe_min[i]]
                unsafe_space.append((A, b))
                A[0][i] = 1.0
                b = [safe_max[i]]
                unsafe_space.append((A, b))
            env = (self.env.A.tolist(), self.env.B.tolist(),
                    self.env.continuous, dt, unsafe_space)
        covers = []
        for inv in self.cover_list:
            covers.append((inv[0].tolist(),
                inv[1].flatten().tolist()[0],
                inv[2].flatten().tolist()[0],
                inv[3].flatten().tolist()[0]))

        controllers = []
        for k in self.K_list:
            controllers.append(k.tolist())

        ret = synthesis.get_covers(env, controllers, covers, bound)

        for (A, b) in ret:
            self.use_list.append((np.matrix(A),
                np.matrix(map(lambda x: [x], b))))

    @timeit
    def train_shield(self, old_shield, actor, bound=20):
        """Train a shield.

        This simply invokes the C++ extension, see synthesis.cpp for a more
        detailed description of the synthesis algorithm. This algorithm
        requires the old shield to use as a starting point for synthesis.

        Arguments:
            old_shield (Shield): The previous shield for this environment.
        """

        dt = self.env.timestep if self.env.continuous else 0.01

        if isinstance(self.env, Environment.PolySysEnvironment):
            unsafe_space = []
            for (A, b) in zip(self.env.unsafe_A, self.env.unsafe_b):
                unsafe_space.append((A.tolist(), np.asarray(b).flatten().tolist()))
            env = (self.env.capsule, self.env.continuous, dt, unsafe_space)
        else:
            # unsafe_space format: [(matrix, vector}]
            unsafe_space = []
            safe_min = self.env.x_min
            safe_max = self.env.x_max
            for i in range(len(safe_min)):
                A = [[0.0] * len(safe_min)]
                A[0][i] = -1.0
                b = [-safe_min[i]]
                unsafe_space.append((A, b))
                A[0][i] = 1.0
                b = [safe_max[i]]
                unsafe_space.append((A, b))
            env = (self.env.A.tolist(), self.env.B.tolist(),
                    self.env.continuous, dt, unsafe_space)

        # We need to compute bounding boxes for these polytopes. The
        # polytopes are represented as a set of linear constraints. In general
        # we can find the maximum or minimum value for a particular dimension
        # i by solving a linear optimization problem with objective x_i or
        # -x_i and the existing constraints.
        covers = []
        for inv in old_shield.cover_list:
            covers.append((inv[0].tolist(),
                inv[1].flatten().tolist()[0],
                inv[2].flatten().tolist()[0],
                inv[3].flatten().tolist()[0]))

        controllers = []
        for k in old_shield.K_list:
            controllers.append(k.tolist())

        def measure(K, space, dataset):
            # TODO: Use dataset to implement DAgger. The dataset
            # object should be created if it is None and updated with
            # new trajectories otherwise.
            A = np.matrix(space[0])
            b = np.matrix(map(lambda x: [x], space[1]))
            lower = np.matrix(map(lambda x: [x], space[2]))
            upper = np.matrix(map(lambda x: [x], space[3]))
            contr = np.matrix(K)
            grad = np.zeros_like(K)
            total = 0.0
            for _ in range(50):
                # sample an initial state from the cover of this controller
                iters = 0
                while True:
                    x = np.random.random_sample(lower.shape)
                    x = lower + np.multiply(x, upper - lower)
                    if (A * x <= b).all():
                        break
                    iters += 1
                    if iters > 100:
                        # This space is very low-density in the region
                        # In this case we will just return some value because
                        # the probability of the state of the system reaching
                        # this space is low
                        return (0.0, dataset)
                    #else:
                    #    print x
                    #    print A * x
                    #    print b
                diff = 0.0
                for _ in range(30):
                    u_n = actor.predict(x.transpose()).T
                    u_k = contr * x
                    diff += np.linalg.norm(u_n - u_k)
                    if isinstance(self.env, Environment.Environment):
                        xp = self.env.A * x + self.env.B * u_k
                    else:
                        xp = self.env.polyf(x, u_k)
                    if self.env.continuous:
                        x = x + self.env.timestep * xp
                    else:
                        x = xp
                total += diff / 30
                grad += (1.0 / 30) * (u_k - u_n) * x.T
            return ((0.01 * grad).tolist(), -total / 100, dataset)

        ret = synthesis.synthesize_shield(env, covers, controllers,
                bound, measure)

        self.K_list = []
        self.inv_list = []
        self.cover_list = []
        for (k, (A, b), (sA, sb, l, u)) in ret:
            self.K_list.append(np.matrix(k))
            self.inv_list.append((np.matrix(A),
                np.matrix(map(lambda x: [x], b))))
            self.cover_list.append((np.matrix(sA),
                np.matrix(map(lambda x: [x], sb)),
                np.matrix(map(lambda x: [x], l)),
                np.matrix(map(lambda x: [x], u))))
        self.set_covers(bound)

    def save_shield(self, model_path):
        """Save a shield to a file.

        Arguments:
            model_path (string): The path to save this shield to.
        """
        # TODO
        raise NotImplementedError("save_shield is not yet implemented")

    def load_shield(self, model_path, enable_jit):
        """Load a shield previous saved with save_shield().

        Arguments:
            model_path (string): The path to load the shield from.
        """
        # TODO
        raise NotImplementedError("load_shield is not yet implemented")

    def detector(self, x, u):
        """Determine whether an action is unsafe under this shield.

        Arguments:
            x (np.matrix): current state
            u (np.matrix): current action

        Returns:
            bool: True if the action is unsafe.
        """
        if isinstance(self.env, Environment.Environment):
            n = self.env.A * x + self.env.B * u
        else:
            n = self.env.polyf(x, u)

        if self.env.continuous:
            n = x + self.env.timestep * n

        for (A, b) in self.inv_list:
            if (A * n <= b).all():
                # We are inside the invariant of some piece of the shield
                self.last_shield = -1
                # print A, b, n, A * n
                return False
        return True

    def call_shield(self, x):
        """Choose an action for a particular state.

        Arguments:
            x (np.matrix): The current state

        Returns:
            np.array: An for the current state.
        """
        if self.last_shield >= 0:
            return self.K_list[self.last_shield] * x

        for i in range(len(self.K_list)):
            (A, b) = self.inv_list[i]
            if (A * x <= b).all():
                self.last_shield = i
                return self.K_list[i] * x
        print x
        for (A, b) in self.inv_list:
            print A
            print b
            print A * x
            print A * x <= b
        raise RuntimeError("No appropriate controller found in shield invocation")

    @timeit
    def train_polysys_shield(self, learning_method, number_of_rollouts,
            simulation_steps, eq_err=1e-2, explore_mag=0.04, step_size=0.05,
            names=None, coffset=None, bias=False, degree=4, aggressive=False,
            without_nn_guide=False, enable_jit=False, nn_weight=0.0):
        """train shield

        Args:
            learning_method (string): learning method string
            number_of_rollouts (int): number of rollouts
            simulation_steps (int): simulation steps
            timestep (float, optional): timestep for continuous control
            eq_err (float, optional): amount of guassian error
            rewardf (None, optional): reward function
            testf (None, optional): reward function for draw controller
            explore_mag (float, optional): explore mag
            step_size (float, optional): step size
            names (None, optional): names of state
        """

        """
        Additional arguments in line 2 of the function signature:
        polyf:                describe polynomial system dynamics in python
        polyf_to_str(K):      describe polynomial system dynamics in string

        rewardf               describe polynomial system reward function
        testf                 describe polynomial system test function

        unsafe_string():      describe polynomial unsafe conditions in string
        """
        self.b_str_list = []
        self.b_list = []
        self.last_b_result = []
        self.b = none
        self.initial_range_list = []
        if self.k_list == []:
            #assert names is not none
            x0 = self.env.reset()

            def learning_oracle_continuous(x):
                self.k = learn_polysys_shield(self.env.polyf,
                        self.env.state_dim, self.env.action_dim, self.env.q,
                        self.env.r, x, eq_err, learning_method,
                        number_of_rollouts, simulation_steps, self.actor,
                        rewardf=self.env.rewardf, continuous=true,
                        timestep=self.env.timestep, explore_mag=explore_mag,
                        step_size=step_size, coffset=coffset, bias=bias,
                        without_nn_guide=without_nn_guide, nn_weight=nn_weight)

                return self.k

            def draw_oracle_continuous(x, k):
                result = test_controller_helper(self.env.polyf, self.k, x,
                        simulation_steps*shield_testing_on_x_ep_len,
                        rewardf=self.env.testf, continuous=true,
                        timestep=self.env.timestep, coffset=coffset, bias=bias)
                if (result >= 0):
                    # find *a new piece of* controller
                    savek(self.model_path, self.k)
                return result

            #iteratively search polcies that can cover all initial states
            def verification_oracle_continuous(x, initial_size, theta, k):
                #theta and k is useless here but required by the api

                #specs for initial conditions
                init = []
                initsospoly = []
                init_cnstr = []
                for i in range(self.env.state_dim):
                    init.append("init" + str(i+1) + " = (x[" + str(i+1) + \
                            "] - " + str(self.env.s_min[i,0]) + ")*(" + \
                            str(self.env.s_max[i,0]) + "-x[" + str(i+1) + \
                            "])")
                for i in range(self.env.state_dim):
                    initsospoly.append("@variable m zinit" + str(i+1) + \
                            " sospoly(z)")
                for i in range(self.env.state_dim):
                    init_cnstr.append(" - zinit" + str(i+1) + "*init" + \
                            str(i+1))
                #specs for initial conditions subject to initial_size
                for i in range(self.env.state_dim):
                    l = x[i,0] - initial_size[i]
                    h = x[i,0] + initial_size[i]
                    init.append("init" + str(self.env.state_dim+i+1) + \
                            " = (x[" + str(i+1) + "] - (" + str(l) + \
                            "))*((" + str(h) + ")-x[" + str(i+1) + "])")
                for i in range(self.env.state_dim):
                    initsospoly.append("@variable m zinit" + \
                            str(self.env.state_dim+i+1) + " sospoly(z)")
                for i in range(self.env.state_dim):
                    init_cnstr.append(" - zinit" + \
                            str(self.env.state_dim+i+1) + "*init" + \
                            str(self.env.state_dim+i+1))

                #specs for unsafe condions
                unsafes = self.env.unsafe_property()
                unsafe = []
                unsafesospoly = []
                unsafe_cnstr = []
                for i in range(len(unsafes)):
                    unsafe.append("unsafe" + str(i+1) + " = " + unsafes[i])
                for i in range(len(unsafes)):
                    unsafesospoly.append("@variable m zunsafe" + str(i+1) + \
                            " sospoly(z)")
                for i in range(len(unsafes)):
                    unsafe_cnstr.append(" - zunsafe" + str(i+1) + \
                            "*unsafe" + str(i+1))

                #specs for bounded state space
                bound = []
                boundsospoly = []
                bound_cnstr = []
                if self.env.bound_x_min is not none and \
                        self.env.bound_x_max is not none:
                    for i in range(self.env.state_dim):
                        if self.env.bound_x_min[i,0] is not none and \
                                self.env.bound_x_max[i,0] is not none:
                            bound.append("bound" + str(i+1) + " = (x[" + \
                                    str(i+1) + "] - " + \
                                    str(self.env.bound_x_min[i,0]) + ")*(" + \
                                    str(self.env.bound_x_max[i,0]) + "-x[" + \
                                    str(i+1) + "])")
                    for i in range(self.env.state_dim):
                        if self.env.bound_x_min[i,0] is not none and \
                                self.env.bound_x_max[i,0] is not none:
                            boundsospoly.append("@variable m zbound" + \
                                    str(i+1) + " sospoly(z)")
                    for i in range(self.env.state_dim):
                        if self.env.bound_x_min[i,0] is not none and \
                                self.env.bound_x_max[i,0] is not none:
                            bound_cnstr.append(" - zbound" + str(i+1) + \
                                    "*bound" + str(i+1))

                #specs for bounded environment disturbance
                disturbance = []
                disturbancesospoly = []
                disturbance_cnstr = []
                if self.env.disturbance_x_min is not none and \
                        self.env.disturbance_x_max is not none:
                    for i in range(self.env.state_dim):
                        if self.env.disturbance_x_min[i,0] is not none and \
                                self.env.disturbance_x_max[i,0] is not none:
                            disturbance.append("disturbance" + str(i+1) + \
                                    " = (d[" + str(i+1) + "] - " + \
                                    str(self.env.disturbance_x_min[i,0]) + \
                                    ")*(" + \
                                    str(self.env.disturbance_x_max[i,0]) + \
                                    "-d[" + str(i+1) + "])")
                    for i in range(self.env.state_dim):
                        if self.env.disturbance_x_min[i,0] is not none and \
                                self.env.disturbance_x_max[i,0] is not none:
                            disturbancesospoly.append(
                                    "@variable m zdisturbance" + str(i+1) + \
                                            " sospoly(d)")
                    for i in range(self.env.state_dim):
                        if self.env.disturbance_x_min[i,0] is not none and \
                                self.env.disturbance_x_max[i,0] is not none:
                            disturbance_cnstr.append(" - zdisturbance" + \
                                    str(i+1) + "*disturbance" + str(i+1))

                # now we have init, unsafe and sysdynamics for verification
                sos = none
                if self.env.bound_x_min is not none and \
                        self.env.bound_x_max is not none:
                    sos = gensoswithbound(self.env.state_dim,
                            ",".join(self.env.polyf_to_str(k)),
                            "\n".join(init), "\n".join(unsafe),
                            "\n".join(bound), "\n".join(initsospoly),
                            "\n".join(unsafesospoly), "\n".join(boundsospoly),
                            "".join(init_cnstr), "".join(unsafe_cnstr),
                            "".join(bound_cnstr), degree=degree)
                elif self.env.disturbance_x_min is not none and \
                        self.env.disturbance_x_max is not none:
                    sos = gensoswithdisturbance(self.env.state_dim,
                            ",".join(self.env.polyf_to_str(k)),
                            "\n".join(init), "\n".join(unsafe),
                            "\n".join(disturbance), "\n".join(initsospoly),
                            "\n".join(unsafesospoly),
                            "\n".join(disturbancesospoly), "".join(init_cnstr),
                            "".join(unsafe_cnstr), "".join(disturbance_cnstr),
                            degree=degree)
                else:
                    sos = gensos(self.env.state_dim,
                            ",".join(self.env.polyf_to_str(k)),
                            "\n".join(init), "\n".join(unsafe),
                            "\n".join(initsospoly), "\n".join(unsafesospoly),
                            "".join(init_cnstr), "".join(unsafe_cnstr),
                            degree=degree)
                #verified = verifysos(writesos("sos.jl", sos), false, 900,
                verified = verifysos(writesos("sos.jl", sos), false, 300,
                        aggressive=aggressive)
                print verified

                #if verified.split("#")[0].find("optimal") >= 0:
                if verified.split("#")[0].find("optimal") >= 0:
                    return true, verified.split("#")[1]
                else:
                    return false, none

            theta = (self.env.s_min, self.env.s_max)
            result, resultlist = verify_controller_z3(x0, theta, verification_oracle_continuous, learning_oracle_continuous, draw_oracle_continuous, continuous=true)
            print ("shield synthesis result: {}".format(result))
            if result:
                for (x, initial_size, inv, k) in resultlist:
                    self.b_str_list.append(inv+"\n")
                    self.b_list.append(barrier_certificate_str2func(
                        inv, self.env.state_dim, enable_jit))
                    self.k_list.append(k)
                    initial_range = np.array(
                            [x - initial_size.reshape(len(initial_size), 1),
                             x + initial_size.reshape(len(initial_size), 1)])
                    self.initial_range_list.append(initial_range)

                self.save_shield(os.path.split(self.model_path)[0])
        else:
            self.load_shield(os.path.split(self.model_path)[0], enable_jit)


    @timeit
    def test_shield(self, actor, test_ep=1, test_step=5000, x0=None, mode="single", loss_compensation=0, shield_combo=1, mute=False):
        """test if shield works

        Args:
            test_ep (int, optional): test episodes
            test_step (int, optional): test step in each episode
        """
        assert shield_combo > 0
        assert loss_compensation >= 0

        fail_time = 0
        success_time = 0
        fail_list = []
        self.shield_count = 0
        combo_remain = 0

        for ep in xrange(test_ep):
            if x0 is not None:
                x = self.env.reset(x0)
            else:
                x = self.env.reset()
            init_x = x
            for i in xrange(test_step):
                u = np.reshape(actor.predict(np.reshape(np.array(x), \
                    (1, actor.s_dim))), (actor.a_dim, 1))

                # safe or not
                if self.detector(x, u) or combo_remain > 0:
                    if combo_remain == 0:
                        combo_remain = shield_combo

                    u = self.call_shield(x)
                    if not mute:
                        print "!shield at step {}".format(i)

                    combo_remain -= 1

                # step
                x, _, terminal = self.env.step(u)

                # success or fail
                if terminal:
                    if np.sum(np.power(self.env.xk, 2)) < self.env.terminal_err:
                        success_time += 1
                    else:
                        fail_time += 1
                        fail_list.append((init_x, x))
                    break

                if i == test_step-1:
                    success_time += 1

            print "----epoch: {} ----".format(ep)
            print 'initial state:\n', init_x, '\nterminal state:\n', x, '\nlast action:\n', self.env.last_u
            print "----step: {} ----".format(i)

        print 'Success: {}, Fail: {}'.format(success_time, fail_time)
        print '#############Fail List:###############'
        for (i, e) in fail_list:
          print 'initial state:\n{}\nend state: \n{}\n----'.format(i, e)

        print 'shield times: {}, shield ratio: {}'.format(self.shield_count, float(self.shield_count)/(test_ep*test_step))


    @timeit
    def shield_boundary(self, sample_ep=500, sample_step=100):
        """sample to find the state bound of shield

        Args:
            sample_ep (int, optional): epsoides
            sample_step (int, optional): step in each epsoide
        """
        max_boundary = np.zeros([self.env.state_dim, 1])
        min_boundary = np.zeros([self.env.state_dim, 1])

        for ep in xrange(sample_ep):
            x = self.env.reset()
            for i in xrange(sample_step):
                u = self.call_shield(x)
                max_boundary, min_boundary = metrics.find_boundary(
                        x, max_boundary, min_boundary)
                # step
                x, _, terminal = self.env.step(u)

        print 'max_boundary:\n{}\nmin_boundary:\n{}'.format(
                max_boundary, min_boundary)

    def learn_shield_gd(self, lr=0.00001, epsoides=100, steps=1000):
        K = np.random.random(self.env.state_dim)
        grad = np.zeros(self.env.state_dim)
        for ep in xrange(epsoides):
            self.env.reset()
            loss = 0
            for step in xrange(steps):
                u = self.actor.predict(np.reshape(np.array(self.env.xk),
                    (1, self.actor.s_dim)))
                grad += np.array(((K.dot(self.env.xk)-u).dot(self.env.xk.T)))[0]
                loss += np.sum(np.power((K.dot(self.env.xk)-u), 2))
                self.env.step(u)
            K -= lr*grad
            print loss

        return K
