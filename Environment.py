################ Environment Module ######################
import numpy as np

#Environment for linear systems
class Environment:
    '''
      Environment for DDPG Algorithm.
    '''
    def __init__(self, A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R,
                    continuous=False, rewardf=None, timestep = 0.01,
                    unsafe=False, unsafe_property=None, multi_boundary=False,
                    bad_reward=-900, terminal_err=0, terminalf=None,
                    unsafe_A=None, unsafe_b=None, ev_min=None, ev_max=None,
                    ev_func=None):

        # State transform matrix
        self.A = A
        self.B = B

        self.unsafe_A = unsafe_A
        self.unsafe_b = unsafe_b

        # initial action space
        self.u_min = u_min
        self.u_max = u_max
        self.action_dim = len(u_min)
        assert len(u_min) == len(u_max)

        # initial state space, s is used to bound the random initial state.
        self.s_min = s_min
        self.s_max = s_max
        self.x_min = x_min
        self.x_max = x_max
        self.multi_boundary = multi_boundary

        self.state_dim = len(s_min)
        assert len(s_min) == len(s_max)
        if x_min is not None and x_max is not None:
            assert len(x_min) == len(x_max)

        self.unsafe = unsafe
        self.unsafe_property = unsafe_property

        # coefficient of reward function
        self.Q = Q
        self.R = R

        # when np.sum(np.power(self.last_u, 2))+np.sum(np.power(self.xk, 2)) < terminal_err, win the game
        self.terminal_err = terminal_err

        # if the model is continuous
        self.continuous = continuous

        # Time step
        self.timestep = timestep

        # reward function
        self.rewardf = rewardf

        # bad reward
        self.bad_reward = bad_reward

        self.terminalf = terminalf

        self.ev_min = ev_min
        self.ev_max = ev_max
        self.ev_func = ev_func

        # sample an initial condition for system
        self.reset()

    def reset(self, x0=None):
        if x0 is None:
            # sample an initial condition for system
            self.x0 = np.matrix(
                [[np.random.uniform(self.s_min[i, 0], self.s_max[i, 0])]
                    for i in range(self.state_dim)],
            )
        else:
            self.x0 = x0
        self.xk = self.x0
        self.last_u = np.zeros((1, self.action_dim))
        if self.ev_min is not None:
            self.extra_vars = np.matrix([[
                np.random.uniform(self.ev_min[i,0], self.ev_max[i,0])]
                  for i in range(len(self.ev_min))])
        return self.xk

    def reward(self, x, u):
        # reward
        if self.rewardf:
            if self.ev_min is not None:
                return self.rewardf(x, u, self.extra_vars)
            else:
                return self.rewardf(x, u)
        else:
            return -(np.sum(self.Q * np.abs(x).reshape([self.state_dim, 1])) \
                    + np.sum(self.R * np.abs(u).reshape([self.action_dim, 1])))

    def step(self, uk, coffset=None, safe=True):
        #uk = np.array([[0]])
        def f(x, u):
            return self.A.dot(x.reshape([self.state_dim, 1])) + \
                    self.B.dot(u.reshape([self.action_dim, 1]))

        self.last_u = uk

        #if (uk > self.u_max).all():
        #    uk = self.u_max
        #elif (uk < self.u_min).all():
        #    uk = self.u_min

        if self.continuous:
            self.xk = self.xk + self.timestep * f(self.xk, uk) \
              if coffset is None \
              else self.xk + self.timestep * (f(self.xk, uk) + coffset)
        else:
            self.xk = f(self.xk, uk)

        if self.ev_func is not None:
            self.extra_vars = self.ev_func(self.xk, uk, self.extra_vars)

        # print self.xk
        # print uk

        return self.observation(safe=safe)

    def observation(self, safe=True):
        xk = self.xk
        reward = self.reward(xk, self.last_u)
        terminal = False
        if self.terminalf is not None:
            if self.ev_min is not None:
                terminal = self.terminalf(xk, self.extra_vars)
            else:
                terminal = self.terminalf(xk)
        if not safe and self.unsafe_A is not None and self.unsafe_b is not None:
            for (A, b) in zip(self.unsafe_A, self.unsafe_b):
                if (A * xk <= b).all():
                    return xk, self.bad_reward, True
        if self.x_max is None and self.x_min is None:
            return xk, reward, terminal
        if not safe and not ((xk < self.x_max).any() and (xk > self.x_min).any()):
            return xk, self.bad_reward, True

        if self.terminalf is None:
            if not self.unsafe:
                # Bad Terminal
                if not ((xk < self.x_max).all() and (xk > self.x_min).all()):
                    terminal = True
                    reward = self.bad_reward
                # Good Terminal
                if np.abs(reward) < self.terminal_err:
                    terminal = True
            else:
                # Bad Terminal
                if self.multi_boundary:
                    if ((np.array(xk) < self.x_max) * \
                            (np.array(xk) > self.x_min)).all(axis=1).any():
                        terminal = True
                        reward = self.bad_reward
                else:
                    if ((np.array(xk) < self.x_max) * \
                            (np.array(xk) > self.x_min)).all():
                        terminal = True
                        reward = self.bad_reward
                # Good Terminal
                if np.abs(reward) < self.terminal_err:
                    print("good terminal")
                    terminal = True


        return xk, reward, terminal

    def simulation(self, uk, coffset=None):
        def f(x, u):
            return self.A.dot(x) + self.B.dot(u)

        if (uk > self.u_max).all():
            uk = self.u_max
        elif (uk < self.u_min).all():
            uk = self.u_min

        if self.continuous:
            xk = self.xk + self.timestep * (f(self.xk, uk)) \
                if coffset is None \
                else self.xk + self.timestep * (f(self.xk, uk) + coffset)
        else:
            xk = f(self.xk, uk)

        return xk


#Environment for Polynomial Systems
class PolySysEnvironment:
    '''
      Environment for DDPG Algorithm.
    '''
    def __init__(self, polyf, polyf_to_str, rewardf, testf, unsafe_property,
                  state_dim, action_dim, Q, R, s_min, s_max,
                  x_min=None, x_max=None, u_min=None, u_max=None,
                  bound_x_min=None, bound_x_max=None, disturbance_x_min=None,
                  disturbance_x_max=None, continuous=True, timestep = 0.01,
                  unsafe=False, multi_boundary=False, bad_reward=-900,
                  terminal_err=0, capsule=None, unsafe_A=None, unsafe_b=None,
                  approx=False, breaks=None, break_breaks=None, lower_As=None,
                  lower_Bs=None, upper_As=None, upper_Bs=None,
                  terminalf=None):

        # system dynamics:
        self.polyf = polyf
        self.polyf_to_str = polyf_to_str
        # reward function:
        self.rewardf = rewardf
        self.testf = testf
        # unsafe property:
        self.unsafe_property = unsafe_property

        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(s_min) == len(s_max)
        assert len(s_min) == state_dim

        self.u_min = u_min
        self.u_max = u_max

        # initial state space, s is used to bound the random initial state.
        self.s_min = s_min
        self.s_max = s_max
        self.x_min = x_min
        self.x_max = x_max
        self.bound_x_min = bound_x_min
        self.bound_x_max = bound_x_max
        self.disturbance_x_min = disturbance_x_min
        self.disturbance_x_max = disturbance_x_max

        self.unsafe = unsafe

        self.terminalf = terminalf

        # coefficient of reward function
        self.Q = Q
        self.R = R

        # when np.sum(np.power(self.last_u, 2))+np.sum(np.power(self.xk, 2)) < terminal_err, win the game
        self.terminal_err = terminal_err

        self.continuous = continuous
        self.timestep = timestep
        self.bad_reward = bad_reward
        self.capsule = capsule
        self.unsafe_A = unsafe_A
        self.unsafe_b = unsafe_b

        self.approx = approx
        self.breaks = breaks
        self.break_breaks = break_breaks
        self.lower_As = lower_As
        self.lower_Bs = lower_Bs
        self.upper_As = upper_As
        self.upper_Bs = upper_Bs

        self.ev_min = None
        self.ev_max = None

        # sample an initial condition for system
        self.reset()

    def reset(self, x0=None):
        if x0 is None:
            # sample an initial condition for system
            self.x0 = np.matrix(
                [[np.random.uniform(self.s_min[i, 0], self.s_max[i, 0])] \
                        for i in range(self.state_dim)],
            )
        else:
            self.x0 = np.matrix(x0)
        self.xk = self.x0
        self.last_u = np.zeros((1, self.action_dim))
        return np.matrix(self.xk)

    def reward(self, x, u):
        if self.rewardf is not None:
            if self.ev_min is not None:
                return self.rewardf(x, self.Q, u, self.R, self.extra_vars)
            else:
                return self.rewardf(x, self.Q, u, self.R)
        else:
            return -(np.sum(self.Q * np.abs(x).reshape([self.state_dim, 1])) \
                    + np.sum(self.R * np.abs(u).reshape([self.action_dim, 1])))

    def step(self, uk, coffset=None, safe=True):
        f = self.polyf

        self.last_u = uk

        if self.continuous:
            self.xk = self.xk + self.timestep * (f(self.xk, uk)) \
              if coffset is None \
              else self.xk + self.timestep * (f(self.xk, uk) + coffset)
        else:
            self.xk = f(self.xk, uk)

        # print self.xk
        # print uk

        return self.observation(safe=safe)

    def observation(self, safe=True):
        xk = self.xk
        reward = self.reward(xk, self.last_u)
        terminal = False
        if self.testf(xk, self.last_u) < 0:
            terminal = True
            reward = self.bad_reward
        if np.abs(reward) < self.terminal_err:
            terminal = True
        if self.terminalf is not None and self.terminalf(xk):
            terminal = True
        if not safe and self.unsafe_A is not None and self.unsafe_b is not None:
            for (A, b) in zip(self.unsafe_A, self.unsafe_b):
                if (A * xk <= b).all():
                    return xk, self.bad_reward, True

        return xk, reward, terminal

    def simulation(self, uk, coffset=None):
        f = self.polyf

        if self.continuous:
            xk = self.xk + self.timestep * (f(self.xk, uk)) \
                if coffset is None \
                else self.xk + self.timestep * (f(self.xk, uk) + coffset)
        else:
            xk = f(self.xk, uk)

        return xk
