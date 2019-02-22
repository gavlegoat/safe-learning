import metrics
from metrics import timeit
from main import *

import os

class Shield(object):

  def __init__(self, env, actor, model_path=None, force_learning=False, debug=False):
    """init
    
    Args:
        env (Environment): environment
        actor (ActorNetwork): actor
        force_learning (bool, optional): if true, even there are model stored in model path, still train.
    """
    self.env = env
    self.actor = actor

    self.model_path = model_path
    self.K = None
    self.K_list = []
    self.initial_range_list = []
    if not force_learning and os.path.isfile(str(self.model_path)):
      self.K_list = [_K for _K in loadK(self.model_path)]
    self.continuous = env.continuous

    self.shield_count = 0

    self.debug = debug

    self.step_count = 0
    self.last_B_value = 0
    self.keep_increasing = False

  @timeit
  def train_shield(self, learning_method, number_of_rollouts, simulation_steps, eq_err=1e-2, rewardf=None, testf=None, explore_mag = .04, step_size = .05, names=None, 
    coffset=None, bias=False, discretization=False, lqr_start=False, degree=4, without_nn_guide=False):
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
    # continuous
    if self.env.continuous:
      self.B_str_list = []
      self.B_list = []
      self.last_B_result = []
      self.B = None
      if self.K_list == []:
        #assert names is not None
        x0 = self.env.reset()

        def default_testf_continous(x, u):
          if self.env.unsafe:
            if ((np.array(x) < self.env.x_max)*(np.array(x) > self.env.x_min)).all(axis=1).any():
              return -1
            else:
              return 0
          else:
            if ((x < self.env.x_max).all() and (x > self.env.x_min).all()):
              return 0
            else:
              return -1 

        def learning_oracle_continuous(x):
          self.K = learn_shield(self.env.A, self.env.B, self.env.Q, self.env.R, x, eq_err,\
            learning_method, number_of_rollouts, simulation_steps, self.actor, self.env.x_min, self.env.x_max, rewardf=rewardf, \
            continuous=True, timestep=self.env.timestep, explore_mag = explore_mag, step_size = step_size, coffset=coffset, bias=bias, \
            unsafe_flag=self.env.unsafe, lqr_start=lqr_start, without_nn_guide=without_nn_guide)
          return self.K

        def draw_oracle_continuous(x, K):
          # draw_controller (self.env.A, self.env.B, self.K, x, simulation_steps*shield_testing_on_x_ep_len, names, True, 0.01)
          test_reward = testf if testf is not None else default_testf_continous
          result = test_controller (self.env.A, self.env.B, self.K, x, simulation_steps*shield_testing_on_x_ep_len, rewardf=test_reward, \
            continuous=True, timestep=self.env.timestep, coffset=coffset, bias=bias)
          return result

        #Iteratively search polcies that can cover all initial states
        '''
        Fixme: the verification approach does not consider the case under which x_min and x_max 
        '''
        def verification_oracle_continuous(x, initial_size, Theta, K):
          #Theta and K is useless here but required by the API

          #Generate the closed loop system for verification
          Acl = self.env.A + self.env.B.dot(self.K)
          print "Learned Closed Loop System: {}".format(Acl)

          if (discretization):
            S0 = Polyhedron.from_bounds(self.env.s_min, self.env.s_max)
            self.O_inf = verify_controller_via_discretization(Acl, self.env.timestep, self.env.x_min, self.env.x_max)
            min = np.array([[x[i,0] - initial_size[i]] for i in range(self.env.state_dim)])
            max = np.array([[x[i,0] + initial_size[i]] for i in range(self.env.state_dim)])
            S = Polyhedron.from_bounds(min, max)
            S = S.intersection(S0)
            ce = S.is_included_in_with_ce(self.O_inf)
            return (ce is None)
          else:
            #Specs for initial conditions
            init = []
            initSOSPoly = []
            init_cnstr = []
            for i in range(self.env.state_dim):
              init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(self.env.s_min[i,0]) + ")*(" + str(self.env.s_max[i,0]) + "-x[" + str(i+1) + "])")    
            for i in range(self.env.state_dim):    
              initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
            for i in range(self.env.state_dim):
              init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))
            #Specs for initial conditions subject to intial_size
            for i in range(self.env.state_dim):
              l = x[i,0] - initial_size[i]
              h = x[i,0] + initial_size[i]
              init.append("init" + str(self.env.state_dim+i+1) + " = (x[" + str(i+1) + "] - (" + str(l) + "))*((" + str(h) + ")-x[" + str(i+1) + "])")    
            for i in range(self.env.state_dim):    
              initSOSPoly.append("@variable m Zinit" + str(self.env.state_dim+i+1) + " SOSPoly(Z)")
            for i in range(self.env.state_dim):
              init_cnstr.append(" - Zinit" + str(self.env.state_dim+i+1) + "*init" + str(self.env.state_dim+i+1))
            #Specs for unsafe condions depends on env.unsafe
            unsafe = []
            unsafeSOSPoly = []
            unsafe_cnstr = []
            if (self.env.unsafe):
              #unsafe is given either via unsafe regions or unsfe properties in the env
              if (self.env.unsafe_property is not None):
                unsafes = self.env.unsafe_property ()
                unsafe = []
                unsafeSOSPoly = []
                unsafe_cnstr = []
                for i in range(len(unsafes)):
                  unsafe.append("unsafe" + str(i+1) + " = " + unsafes[i])
                  unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
                  unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))
              if (self.env.x_min is not None):
                for j in range(len(self.env.x_min)):
                  unsafe_query = ""
                  unsafe_x_min = self.env.x_min[j]
                  unsafe_x_max = self.env.x_max[j]
                  for i in range(self.env.state_dim):
                    if unsafe_x_min[i, 0] != np.NINF and unsafe_x_max[i, 0] != np.inf:
                      unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(unsafe_x_min[i,0]) + ")*(" + str(unsafe_x_max[i,0]) + "-x[" + str(i+1) + "])")    
                      unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
                      unsafe_query += " - Zunsafe" + str(i+1) + "*unsafe" + str(i+1)
                    elif unsafe_x_min[i, 0] != np.NINF:
                      unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(unsafe_x_min[i,0]) + ")*(" + str(unsafe_x_max[i,0]) + "-x[" + str(i+1) + "])")    
                      unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
                      unsafe_query += " - Zunsafe" + str(i+1) + "*unsafe" + str(i+1)
                    elif unsafe_x_max[i, 0] != np.inf: 
                      unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(unsafe_x_min[i,0]) + ")*(" + str(unsafe_x_max[i,0]) + "-x[" + str(i+1) + "])")    
                      unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
                      unsafe_query += " - Zunsafe" + str(i+1) + "*unsafe" + str(i+1)
                  if unsafe_query != "":
                    unsafe_cnstr.append(unsafe_query)
            else:
              for i in range(self.env.state_dim):
                mid = (self.env.x_min[i, 0] + self.env.x_max[i, 0]) / 2
                radium = self.env.x_max[i, 0] - mid
                unsafe.append("unsafe" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(mid) + ")^2 - " + str(pow(radium, 2)))    
                unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
                unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))
             # Now we have init, unsafe and sysdynamics for verification
            sos = genSOSContinuousAsDiscreteMultipleUnsafes(
                        self.env.timestep, self.env.state_dim, ",".join(dxdt(Acl)), "\n".join(init), "\n".join(unsafe), 
                        "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "".join(init_cnstr), unsafe_cnstr, degree=degree)
            verified = verifySOS(writeSOS("SOS.jl", sos), False, 900)
            print verified
            
            if verified.split("#")[0].find("Optimal") >= 0:
              # returns Verified and the inductive invariant
              return True, verified.split("#")[1]
            else:
              return False, None
            #return (verified.find("Optimal") >= 0)

        Theta = (self.env.s_min, self.env.s_max)
        result, resultList = verify_controller_z3(x0, Theta, verification_oracle_continuous, learning_oracle_continuous, draw_oracle_continuous, continuous=True)
        print ("Shield synthesis result: {}".format(result))
        if result:
          for (x, initial_size, inv, K) in resultList:
            self.B_str_list.append(inv+"\n")
            self.B_list.append(barrier_certificate_str2func(inv, self.env.state_dim))
            self.K_list.append(K)
            initial_range = np.array([x-initial_size.reshape(len(initial_size), 1), x+initial_size.reshape(len(initial_size), 1)])
            self.initial_range_list.append(initial_range)

          self.save_shield(os.path.split(self.model_path)[0])
      else:
        self.load_shield(os.path.split(self.model_path)[0])

    # discrete
    else:
      self.O_inf_list = []
      self.last_O_inf_result = []
      self.O_inf = None
      if self.K_list == []:
        x0 = self.env.reset()
        S0 = Polyhedron.from_bounds(self.env.s_min, self.env.s_max)

        def default_testf_discrete(x, u):
          if self.env.unsafe:
            if ((np.array(x) < self.env.x_max)*(np.array(x) > self.env.x_min)).all(axis=1).any():
              return -1
            else:
              return 0
          else:
            if ((x < self.env.x_max).all() and (x > self.env.x_min).all()) and ((u < self.env.u_max).all() and (u > self.env.u_min).all()):
              return 0
            else:
              return -1

        def learning_oracle_discrete(x):
          self.K = learn_shield(self.env.A, self.env.B, self.env.Q, self.env.R, x, eq_err,\
            learning_method, number_of_rollouts, simulation_steps, self.actor, self.env.x_min, self.env.x_max, rewardf=rewardf,\
            continuous=False, timestep=self.env.timestep, explore_mag = explore_mag, step_size = step_size, coffset=coffset, bias=bias, \
            unsafe_flag=self.env.unsafe, lqr_start=lqr_start, without_nn_guide=without_nn_guide)
          return self.K

        def draw_oracle_discrete(x, K):
          # draw_controller (self.env.A, self.env.B, self.K, x, simulation_steps*shield_testing_on_x_ep_len, names, True, 0.01)
          test_reward = testf if testf is not None else default_testf_discrete
          result = test_controller (self.env.A, self.env.B, self.K, x, simulation_steps*shield_testing_on_x_ep_len, rewardf=test_reward, \
            coffset=coffset, bias=bias)
          return result

        #Iteratively search polcies that can cover all initial states
        def verification_oracle_discrete(x, initial_size, Theta, K):
          self.O_inf = verify_controller(np.asarray(self.env.A), np.asarray(self.env.B), np.asarray(self.K), self.env.x_min, self.env.x_max, self.env.u_min, self.env.u_max)
          min = np.array([[x[i,0] - initial_size[i]] for i in range(self.env.state_dim)])
          max = np.array([[x[i,0] + initial_size[i]] for i in range(self.env.state_dim)])

          S = Polyhedron.from_bounds(min, max)
          S = S.intersection(S0)
          ce = S.is_included_in_with_ce(self.O_inf)
          if ce is None:
            self.K_list.append(K)
            self.O_inf_list.append(self.O_inf)
            initial_range = np.array([x-initial_size.reshape(len(initial_size), 1), x+initial_size.reshape(len(initial_size), 1)])
            self.initial_range_list.append(initial_range)
          return (ce is None)  

        Theta = (self.env.s_min, self.env.s_max)
        result = verify_controller_z3(x0, Theta, verification_oracle_discrete, learning_oracle_discrete, draw_oracle_discrete, continuous=False)
        print ("Shield synthesis result: {}".format(result))  
        if result:
          self.save_shield(os.path.split(self.model_path)[0])        
      else:
        self.load_shield(os.path.split(self.model_path)[0])




  @timeit
  def train_polysys_shield(self, learning_method, number_of_rollouts, simulation_steps, eq_err=1e-2, 
        explore_mag = .04, step_size = .05, names=None, coffset=None, bias=False, degree=4, aggressive=False, without_nn_guide=False):
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
    self.B_str_list = []
    self.B_list = []
    self.last_B_result = []
    self.B = None
    self.initial_range_list = []
    if self.K_list == []:
      #assert names is not None
      x0 = self.env.reset()

      def learning_oracle_continuous(x):
        self.K = learn_polysys_shield(self.env.polyf, self.env.state_dim, self.env.action_dim, self.env.Q, self.env.R, x, eq_err,\
          learning_method, number_of_rollouts, simulation_steps, self.actor, rewardf=self.env.rewardf, \
          continuous=True, timestep=self.env.timestep, explore_mag = explore_mag, step_size = step_size, coffset=coffset, bias=bias, without_nn_guide=without_nn_guide)

        return self.K

      def draw_oracle_continuous(x, K):
        result = test_controller_helper(self.env.polyf, self.K, x, simulation_steps*shield_testing_on_x_ep_len, rewardf=self.env.testf, continuous=True, timestep=self.env.timestep,\
          coffset=coffset, bias=bias)
        if (result >= 0):
          # Find *a new piece of* controller
          saveK(self.model_path, self.K)
        return result

        #Iteratively search polcies that can cover all initial states
      def verification_oracle_continuous(x, initial_size, Theta, K):
        #Theta and K is useless here but required by the API

        #Specs for initial conditions
        init = []
        initSOSPoly = []
        init_cnstr = []
        for i in range(self.env.state_dim):
          init.append("init" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(self.env.s_min[i,0]) + ")*(" + str(self.env.s_max[i,0]) + "-x[" + str(i+1) + "])")    
        for i in range(self.env.state_dim):    
          initSOSPoly.append("@variable m Zinit" + str(i+1) + " SOSPoly(Z)")
        for i in range(self.env.state_dim):
          init_cnstr.append(" - Zinit" + str(i+1) + "*init" + str(i+1))
        #Specs for initial conditions subject to initial_size
        for i in range(self.env.state_dim):
          l = x[i,0] - initial_size[i]
          h = x[i,0] + initial_size[i]
          init.append("init" + str(self.env.state_dim+i+1) + " = (x[" + str(i+1) + "] - (" + str(l) + "))*((" + str(h) + ")-x[" + str(i+1) + "])")    
        for i in range(self.env.state_dim):    
          initSOSPoly.append("@variable m Zinit" + str(self.env.state_dim+i+1) + " SOSPoly(Z)")
        for i in range(self.env.state_dim):
          init_cnstr.append(" - Zinit" + str(self.env.state_dim+i+1) + "*init" + str(self.env.state_dim+i+1))
        
        #Specs for unsafe condions
        unsafes = self.env.unsafe_property()
        unsafe = []
        unsafeSOSPoly = []
        unsafe_cnstr = []
        for i in range(len(unsafes)):
          unsafe.append("unsafe" + str(i+1) + " = " + unsafes[i])
        for i in range(len(unsafes)):
          unsafeSOSPoly.append("@variable m Zunsafe" + str(i+1) + " SOSPoly(Z)")
        for i in range(len(unsafes)):
          unsafe_cnstr.append(" - Zunsafe" + str(i+1) + "*unsafe" + str(i+1))

        #Specs for bounded state space
        bound = []
        boundSOSPoly = []
        bound_cnstr = []
        if (self.env.bound_x_min is not None and self.env.bound_x_max is not None):
          for i in range(self.env.state_dim):
            if (self.env.bound_x_min[i,0] is not None and self.env.bound_x_max[i,0] is not None):
              bound.append("bound" + str(i+1) + " = (x[" + str(i+1) + "] - " + str(self.env.bound_x_min[i,0]) + ")*(" + str(self.env.bound_x_max[i,0]) + "-x[" + str(i+1) + "])")    
          for i in range(self.env.state_dim):
            if (self.env.bound_x_min[i,0] is not None and self.env.bound_x_max[i,0] is not None):    
              boundSOSPoly.append("@variable m Zbound" + str(i+1) + " SOSPoly(Z)")
          for i in range(self.env.state_dim):
            if (self.env.bound_x_min[i,0] is not None and self.env.bound_x_max[i,0] is not None):
              bound_cnstr.append(" - Zbound" + str(i+1) + "*bound" + str(i+1))

        #Specs for bounded environment disturbance
        disturbance = []
        disturbanceSOSPoly = []
        disturbance_cnstr = []
        if (self.env.disturbance_x_min is not None and self.env.disturbance_x_max is not None):
          for i in range(self.env.state_dim):
            if (self.env.disturbance_x_min[i,0] is not None and self.env.disturbance_x_max[i,0] is not None):
              disturbance.append("disturbance" + str(i+1) + " = (d[" + str(i+1) + "] - " + str(self.env.disturbance_x_min[i,0]) + ")*(" + str(self.env.disturbance_x_max[i,0]) + "-d[" + str(i+1) + "])")    
          for i in range(self.env.state_dim):
            if (self.env.disturbance_x_min[i,0] is not None and self.env.disturbance_x_max[i,0] is not None):    
              disturbanceSOSPoly.append("@variable m Zdisturbance" + str(i+1) + " SOSPoly(D)")
          for i in range(self.env.state_dim):
            if (self.env.disturbance_x_min[i,0] is not None and self.env.disturbance_x_max[i,0] is not None):
              disturbance_cnstr.append(" - Zdisturbance" + str(i+1) + "*disturbance" + str(i+1)) 

        # Now we have init, unsafe and sysdynamics for verification
        sos = None
        if (self.env.bound_x_min is not None and self.env.bound_x_max is not None):
          sos = genSOSwithBound(self.env.state_dim, ",".join(self.env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(bound),
                    "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "\n".join(boundSOSPoly),
                    "".join(init_cnstr), "".join(unsafe_cnstr), "".join(bound_cnstr), degree=degree)
        elif (self.env.disturbance_x_min is not None and self.env.disturbance_x_max is not None):
          sos = genSOSwithDisturbance(self.env.state_dim, ",".join(self.env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe), "\n".join(disturbance),
                    "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly), "\n".join(disturbanceSOSPoly),
                    "".join(init_cnstr), "".join(unsafe_cnstr), "".join(disturbance_cnstr), degree=degree)
        else:
          sos = genSOS(self.env.state_dim, ",".join(self.env.polyf_to_str(K)), "\n".join(init), "\n".join(unsafe),
                    "\n".join(initSOSPoly), "\n".join(unsafeSOSPoly),
                    "".join(init_cnstr), "".join(unsafe_cnstr), degree=degree)
        verified = verifySOS(writeSOS("SOS.jl", sos), False, 900, aggressive=aggressive)
        print verified

        if verified.split("#")[0].find("Optimal") >= 0:
          return True, verified.split("#")[1]
        else:
          return False, None

      Theta = (self.env.s_min, self.env.s_max)
      result, resultList = verify_controller_z3(x0, Theta, verification_oracle_continuous, learning_oracle_continuous, draw_oracle_continuous, continuous=True)
      print ("Shield synthesis result: {}".format(result))
      if result:
        for (x, initial_size, inv, K) in resultList:
            self.B_str_list.append(inv+"\n")
            self.B_list.append(barrier_certificate_str2func(inv, self.env.state_dim))
            self.K_list.append(K)
            initial_range = np.array([x-initial_size.reshape(len(initial_size), 1), x+initial_size.reshape(len(initial_size), 1)])
            self.initial_range_list.append(initial_range)

        self.save_shield(os.path.split(self.model_path)[0])
    else:
      self.load_shield(os.path.split(self.model_path)[0])

  def save_shield(self, model_path):
    if self.env.continuous:
      with open(model_path+"/shield.model", "w") as f:
        for B_str in self.B_str_list:
          f.write(B_str)
          # print B_str
      print "store shield to "+model_path+"/shield.model"
      saveK(model_path+"/K.model", np.array(self.K_list))
      print "store K to "+model_path+"/K.model.npy"
      saveK(model_path+"/initial_range.model", np.array(self.initial_range_list))
      print "store initial_range to "+model_path+"/initial_range.model.npy"
    else:
      saveK(model_path+"/K.model", np.array(self.K_list))
      print "store K to "+model_path+"/K.model.npy"
      saveK(model_path+"/initial_range.model", np.array(self.initial_range_list))
      print "store initial_range to "+model_path+"/initial_range.model.npy"

  def load_shield(self, model_path):
    if self.env.continuous:
      with open(model_path+"/shield.model", "r") as f:
        for B_str in f:
          self.B_list.append(barrier_certificate_str2func(B_str, self.env.state_dim))
      print "load barrier from " + model_path + "/shield.model"
      self.K_list = [K for K in loadK(model_path+"/K.model.npy")]
      print "load K from "+model_path+"/K.model.npy"
      self.initial_range_list = [initr for initr in loadK(model_path+"/initial_range.model.npy")] 
      print "load initial range to "+model_path+"/initial_range.model.npy"
    else:
      self.K_list = [K for K in loadK(model_path+"/K.model.npy")]
      print "load K from "+model_path+"/K.model.npy"
      self.initial_range_list = [initr for initr in loadK(model_path+"/initial_range.model.npy")] 
      print "load initial range to "+model_path+"/initial_range.model.npy"
      for K in self.K_list:
          O_inf = verify_controller(np.asarray(self.env.A), np.asarray(self.env.B), np.asarray(K), self.env.x_min, self.env.x_max, self.env.u_min, self.env.u_max)
          self.O_inf_list.append(O_inf)

  def select_shield(self):
    i = -1

    if (len(self.initial_range_list) > 1):
      lowboundaries = np.array([item[0] for item in self.initial_range_list])
      upboundaries = np.array([item[1] for item in self.initial_range_list])
      if self.debug:
        print "x0: \n", self.env.x0
        print "low boundary: \n", lowboundaries
        print "up boundary: \n", upboundaries
      select_list = [(self.env.x0>low).all()*(self.env.x0<high).all() for low, high in zip(lowboundaries, upboundaries)]
      i = select_list.index(True)
      if self.debug:
        print "select list", select_list
    elif (len(self.initial_range_list) == 1):
      i == 0    
    else:
      print "Error: No shield available!"
      assert (False)

    self.K = self.K_list[i]

    if self.continuous:
      self.B = self.B_list[i]
      return self.B
    else:
      self.O_inf = self.O_inf_list[i]
      return self.O_inf


  def detactor(self, x, u, mode="single", loss_compensation=0.0, increase_step=-1):
    """detact if there are dangerous state in furture
    
    Args:
        x: current state
        u: current action
        mode (str, optional): single(faster, more calls) -> choose one shield according to the initial state.
                              all(slower, less calls) -> use all shield at run time, if all the B > 0, call shield.
        loss_compensation (float, optional): The compensation for loss in calculating barrier
        increase_step (int, optional): if B's value keep increase this step, call shield until the vale stop increasing,
                                       now only support the single mode.
    
    Returns:
        Bool: True -> call shield
              False -> call neural network
    """
    mode_tuple = ("single", "all")
    assert mode in mode_tuple

    xk = self.env.simulation(u)
    # single shield model
    if mode == mode_tuple[0]:
      # continuous
      if self.env.continuous:
        if self.B is None:
          self.select_shield()
        B_value = self.B(*state2list(xk))

        if self.debug:
          print B_value

        if increase_step >= 0:
          if B_value > self.last_B_value:
            self.step_count += 1
          else:
            self.keep_increasing = False
          self.last_B_value = B_value
          if self.step_count >= increase_step:
            self.step_count = 0
            self.keep_increasing = True
          if self.keep_increasing:
            return True

        if B_value > -loss_compensation:
          return True
        return False
      # discrete
      else:
        self.select_shield()
        if self.O_inf.contains(xk):
          return False
        return True
    # all shield model
    elif mode == mode_tuple[1]:
      # continuous
      if self.env.continuous:
        current_B_result = []
        if self.last_B_result == []:
          lowboundaries = np.array([i[0] for i in self.initial_range_list])
          upboundaries = np.array([i[1] for i in self.initial_range_list])
          self.last_B_result = [np.logical_not((self.env.x0>low).all()*(self.env.x0<high).all()) for low, high in zip(lowboundaries, upboundaries)]
        debug_list = []
        for B in self.B_list:
          B_value = B(*state2list(xk))
          if self.debug:
            debug_list.append(B_value)
          res = B_value > -loss_compensation
          current_B_result.append(res)
        if self.debug:
          print debug_list

        if np.array(current_B_result).all():
          # The K will be called latter
          self.K = self.K_list[self.last_B_result.index(False)]
          return True

        self.last_B_result = current_B_result
        return False
      # discrete
      else:
        current_O_inf_result = []
        if self.last_O_inf_result == []:
          lowboundaries = np.array([i[0] for i in self.initial_range_list])
          upboundaries = np.array([i[1] for i in self.initial_range_list])
          self.last_O_inf_result = [np.logical_not((self.env.x0>low).all()*(self.env.x0<high).all()) for low, high in zip(lowboundaries, upboundaries)]
          
        for O_inf in self.O_inf_list:
          res = not O_inf.contains(xk)
          current_O_inf_result.append(res)
        if self.debug:
          print xk
          print current_O_inf_result

        if np.array(current_O_inf_result).all():
          # The K will be called latter
          self.K = self.K_list[self.last_O_inf_result.index(False)]
          return True

        self.last_O_inf_result = current_O_inf_result
        return False


  def call_shield(self, x, mute=False):
    """call shield
    
    Args:
        x : current state
        mute (bool, optional): print !shield or not
    
    Returns:  
        shield action
    """
    u = self.K.dot(x)
    if not mute:
      print 'Shield! in state: \n', x
    self.shield_count += 1

    return u
      

  @timeit
  def test_shield(self, test_ep=1, test_step=5000, x0=None, mode="single", loss_compensation=0):
    """test if shield works
    
    Args:
        test_ep (int, optional): test episodes
        test_step (int, optional): test step in each episode
    """
    fail_time = 0
    success_time = 0
    fail_list = []
    self.shield_count = 0

    for ep in xrange(test_ep):
      if x0 is not None:
        x = self.env.reset(x0)
      else:
        x = self.env.reset()
      init_x = x
      for i in xrange(test_step):
        u = np.reshape(self.actor.predict(np.reshape(np.array(x), \
            (1, self.actor.s_dim))), (self.actor.a_dim, 1))
        
        # safe or not
        if self.detactor(x, u, mode=mode, loss_compensation=loss_compensation):
          u = self.call_shield(x)

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
        u = self.call_shield(x, mute=True)
        max_boundary, min_boundary = metrics.find_boundary(x, max_boundary, min_boundary)
        # step
        x, _, terminal = self.env.step(u)

    print 'max_boundary:\n{}\nmin_boundary:\n{}'.format(max_boundary, min_boundary)

  def learn_shield_gd(self, lr=0.00001, epsoides=100, steps=1000):
    K = np.random.random(self.env.state_dim)
    grad = np.zeros(self.env.state_dim)
    for ep in xrange(epsoides):
      self.env.reset()
      loss = 0
      for step in xrange(steps):
        u = self.actor.predict(np.reshape(np.array(self.env.xk), (1, self.actor.s_dim)))
        grad += np.array(((K.dot(self.env.xk)-u).dot(self.env.xk.T)))[0]
        loss += np.sum(np.power((K.dot(self.env.xk)-u), 2))
        self.env.step(u)
      K -= lr*grad
      print loss

    return K




import re
def barrier_certificate_str2func(bc_str, vars_num):
  """transform julia barrier string to function
  
  Args:
      bc_str (str): string
      vars_num (int): the dimension number of state
  """
  eval_str = re.sub("\^", r"**", bc_str)
  variables = ["x"+str(i+1) for i in xrange(vars_num)]

  var_pattern = re.compile(r"(?P<var>x\d*)")
  eval_str = var_pattern.sub(r'*\g<var>', eval_str)

  # This way is much much slower
  # def B(state):
  #   values_name=get_values_name(len(state))
  #   assert len(variables) == len(values_name)
  #   eval_str1 = eval_str
  #   for var, val in zip(variables, values_name):
  #     eval_str1 = re.sub(var, val, eval_str1)
  #   return eval(eval_str1)

  args_str = ""
  for arg in variables:
    args_str += (arg+",")
  args_str = args_str[:-1]
  exec("""def B({}): return {}""".format(args_str, eval_str)) in locals()

  return B

def barrier_certificate_str2z3(bc_str, vars_num):
  """transform julia barrier string to what z3 and python can understand
  
  Args:
      bc_str (str): string
  """
  eval_str = re.sub("\^", r"**", bc_str)

  var_pattern = re.compile(r"(?P<var>x\d*)")
  eval_str = var_pattern.sub(r'*\g<var>', eval_str)

  # substitute x1 to x[0], ..., x[n] to x[n-1]
  for i in range(vars_num):
    eval_str = eval_str.replace("x"+str(i+1), "x[" + str(i) + "]")
  # polynomial function's value should be less than 0.
  eval_str = eval_str + " <= 0"

  return eval_str

def get_values_name(vars_num):
  return ["state["+str(i)+"][0]" for i in xrange(vars_num)]

def state2list(state):
  return [x[0] for x in state.tolist()]

