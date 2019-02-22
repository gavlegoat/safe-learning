#!/usr/bin/python

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from pympc.geometry.polyhedron import Polyhedron
from pympc.dynamics.discrete_time_systems import LinearSystem
from pympc.plot import plot_state_space_trajectory

from vcsos import *
from z3verify import verify_controller_z3,bounded_z3

import scipy.linalg as la
from pympc.dynamics.discrete_time_systems import mcais

import os
import time
import random
import subprocess
import platform

from threading import Timer
from metrics import timeit

def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    return -K

def lqr_gain(A,B,Q,R):
  '''
  Arguments:
    State transition matrices (A,B)
    LQR Costs (Q,R)
  Outputs:
    K: optimal infinite-horizon LQR gain matrix given
  '''

  # solve DARE:
  M=scipy.linalg.solve_discrete_are(A,B,Q,R)

  # K=(B'MB + R)^(-1)*(B'MA)
  K = np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))
  return -K

def uniform_random_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,
  linf_norm=3):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters
          linf_norm = maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by uniformly sampling policies
      in bounded region
  '''

  d,p = B.shape

  #### "ALGORITHM"
  best_K = np.empty((p,d))
  best_reward = -float("inf")
  for k in range(N):
    K = np.random.uniform(-linf_norm,linf_norm,(p,d))
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K,x)
      if continuous:
        x = x + timestep*(A.dot(x)+B.dot(u))+eq_err*np.random.randn(d,1)
      else:
        x = A.dot(x)+B.dot(u)+eq_err*np.random.randn(d,1)
      reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
      # Penality added to states 
      if (x_min is not None):
        for index in range(d):
          if x[index, 0] < x_min[index, 0]:
            reward = reward-100
      if (x_max is not None):
        for index in range(d):
          if x[index, 0] > x_max[index, 0]:
            reward = reward-100
    if reward>best_reward:
        best_reward = reward
        best_K = K

  return best_K

def random_search_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 4, coffset=None, bias=False, unsafe_flag=False, lqr_start=False):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters:
        explore_mag = magnitude of the noise to explore
        step_size
        batch_size = number of directions per minibatches
        safeguard: maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by random search
  '''
  def f (x, u):
    return A.dot(x)+B.dot(u)

  d,p = B.shape

  return random_search_helper(f, d, p, Q, R, x0, eq_err, N, T, x_min, x_max, continuous, timestep, rewardf, 
          explore_mag, step_size, batch_size, coffset, bias, unsafe_flag, 
          A if lqr_start and not bias else None, 
          B if lqr_start and not bias else None)

def random_search_helper(f,d,p,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 4, coffset=None, bias=False, unsafe_flag=False, A=None, B=None):
  
  def policy_test(K):
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K, np.vstack([x,[1]])) if bias else np.dot(K, x)
      # Use discrete or continuous semantics based on user's choice
      if continuous:
        x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1) 
      else:
        x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
      if rewardf is None:
        reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
      else:
        reward += rewardf (x, Q, u, R)
      #reward += np.array([[0]])
      # Penality added to states
      if unsafe_flag:
        if ((np.array(x) < x_max)*(np.array(x) > x_min)).all(axis=1).any():
          reward[0,0] = reward[0,0]-100
      else:
        if (x_min is not None):
          for index in range(d):
            if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
              reward[0,0] = reward[0,0]-100
        if (x_max is not None):
          for index in range(d):
            if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
              reward[0,0] = reward[0,0]-100
    return reward

  # initial condition for K
  K0 = 0*np.random.randn(p,d+1) if bias else 0*np.random.randn(p,d)
  if (A is not None and B is not None):
    if (continuous):
      X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))   
      K0 = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    else:
      K0 = dlqr(A, B, Q, R)
  ###

  #### ALGORITHM
  K = K0
  best_K = K
  best_reward = -float("inf")
  for k in range(N):
    reward_store = []
    mini_batch = np.zeros((p,d+1)) if bias else np.zeros((p,d))
    for j in range(batch_size):
      V = np.random.randn(p,d+1) if bias else np.random.randn(p,d)
      for sign in [-1,1]:
        x = x0
        reward = 0
        for t in range(T):
          u = np.dot(K+sign*explore_mag*V,np.vstack([x,[1]])) if bias else np.dot(K+sign*explore_mag*V, x)
          # Use discrete or continuous semantics based on user's choice
          if continuous:
            x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1)
          else:
            x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
          if rewardf is None:
            reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
          else:
            reward += rewardf (x, Q, u, R)
          #reward += np.array([[0]])
          # Penality added to states 
          #safe = True
          unsafe = False
          if unsafe_flag:
            if ((np.array(x) < x_max)*(np.array(x) > x_min)).all(axis=1).any():
              reward[0,0] = reward[0,0]-100
          else:
            if (x_min is not None):
              for index in range(d):
                if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
                  reward[0,0] = reward[0,0]-100
                  #safe = False
                  #print ("unsafe state {}".format(x[index, 0]))
            if (x_max is not None):
              for index in range(d):
                if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
                  reward[0,0] = reward[0,0]-100
          # Break the closed loop system variables are so large
          for index in range(d):
            if abs(x[index, 0]) > 1e72:
              unsafe = True
              break
          if unsafe:
            print ("unsafe x : {} at time {}".format(x, t))
            break
        mini_batch += (reward[0,0]*sign)*V
        reward_store.append(reward)
    #print "reward = {}".format(reward_store) 
    std = np.std(reward_store)
    if (std == 0):
      #More thoughts into this required: K already converged?
      #print ("K seems converged!")
      #return K
      K = K
    else:
      #print ("K is unconverged!")
      #if (np.sum(reward_store) > best_reward):
      #  best_reward = np.sum(reward_store)
      #  best_K = K
      K += (step_size/std/batch_size)*mini_batch
      r = policy_test(K)
      if (r > best_reward):
        best_reward = r
        best_K = K

  #return K
  return best_K

def policy_gradient_adam_linear_policy(A,B,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 8,
    beta1=0.9, beta2=0.999, epsilon=1.0e-8, coffset=None,bias=False):
  '''
    Arguments:
      state transition matrices (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics eq_err
      Number of rollouts N
      Time Horizon T

      hyperparameters
         explore_mag magnitude of the noise to explore
         step_size
         batch_size: number of stochastic gradients per minibatch
         beta1, beta2, epsilon are the additional paramters of Adam

    Outputs:
      Static Control Gain K optimized on LQR cost by Policy Gradient
  '''

  def f (x, u):
    return A.dot(x)+B.dot(u)

  d,p = B.shape

  return policy_gradient_helper(f, d, p, Q, R, x0, eq_err, N, T, x_min, x_max, continuous, timestep, rewardf, 
    explore_mag, step_size, batch_size, 
    beta1, beta2, epsilon, coffset, bias)


def policy_gradient_helper(f,d,p,Q,R,x0,eq_err,N,T,x_min=None,x_max=None,continuous=False,timestep=.01,rewardf=None,
    explore_mag = 0.04, step_size = 0.05, batch_size = 8,
    beta1=0.9, beta2=0.999, epsilon=1.0e-8, coffset=None, bias=False):


  def policy_test(K):
    x = x0
    reward = 0
    for t in range(T):
      u = np.dot(K, x)
      # Use discrete or continuous semantics based on user's choice
      if continuous:
        x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1) 
      else:
        x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
      if rewardf is None:
        reward += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
      else:
        reward += rewardf (x, Q, u, R)
      #reward += np.array([[0]])
      # Penality added to states 
      if (x_min is not None):
        for index in range(d):
          if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
            reward[0,0] = reward[0,0]-100
      if (x_max is not None):
        for index in range(d):
          if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
            reward[0,0] = reward[0,0]-100
    return reward

  # initial condition for K
  K0 = 0.0*np.random.randn(p,d)
  ###

  #### ALGORITHM
  K = K0
  best_K = K
  best_reward = -float("inf")

  baseline = 0.0
  Adam_M = np.zeros((p,d))
  Adam_V = np.zeros((p,d))

  for k in range(N):
    mini_batch = np.zeros((p,d))
    mb_store = np.zeros((p,d,batch_size))
    reward = np.zeros((batch_size))

    # Collect policy gradients for the current minibatch
    for j in range(batch_size):
      x = x0
      X_store = np.zeros((d,T))
      V_store = np.zeros((p,T))
      for t in range(T):
        v = explore_mag*np.random.randn(p,1)
        X_store[:,t] = x.flatten()
        V_store[:,t] = v.flatten()
        u = np.dot(K,x)+v

        # Use discrete or continuous semantics based on user's choice
        if continuous:
          x = x + timestep*(f(x,u))+eq_err*np.random.randn(d,1) if coffset is None else x + timestep*(f(x,u)+coffset)+eq_err*np.random.randn(d,1)
        else:
          x = f(x,u)+eq_err*np.random.randn(d,1) if coffset is None else f(x,u)+coffset+eq_err*np.random.randn(d,1)
        if rewardf is None:
          reward[j] += -np.dot(x.T,Q.dot(x))-np.dot(u.T,R.dot(u))
        else:
          reward[j] += rewardf (x, Q, u, R)
        #reward += np.array([[0]])
        # Penality added to states 
        #safe = True
        unsafe = False
        if (x_min is not None):
          for index in range(d):
            if x_min[index, 0] is not None and x[index, 0] < x_min[index, 0]:
              reward[j] = reward[j]-100
              #safe = False
              #print ("unsafe state {}".format(x[index, 0]))
        if (x_max is not None):
          for index in range(d):
            if x_max[index, 0] is not None and x[index, 0] > x_max[index, 0]:
              reward[j] = reward[j]-100
              #safe = False
              #print ("unsafe state {}".format(x[index, 0]))
        #if ((x_min is not None or x_max is not None) and safe):
        #    reward[0, 0] = reward[0,0] + 100
        #if safe is False:
          #print ("unsafe x : {} at time {}".format(x, t))
          #break
        # Break the closed loop system variables are so large
        for index in range(d):
          if abs(x[index, 0]) > 1e72:
            unsafe = True
            break
        if unsafe:
          print ("unsafe x : {} at time {}".format(x, t))
          break
      mb_store[:,:,j] = np.dot(V_store,X_store.T)

    # Mean of rewards over a minibatch are subtracted from reward.
    # This is a heuristic for baseline subtraction. 

    #print "reward = {}".format(reward)

    for j in range(batch_size):
      mini_batch += ((reward[j]-baseline)/batch_size)*mb_store[:,:,j]
    baseline = np.mean(reward)

    # Adam Algorithm

    Adam_M = beta1*Adam_M + (1-beta1)*mini_batch
    Adam_V = beta2*Adam_V + (1-beta2)*(mini_batch*mini_batch)

    effective_step_size = step_size*np.sqrt(1-beta2**(k+1))/(1-beta1**(k+1))
    K += effective_step_size*Adam_M/(np.sqrt(Adam_V)+epsilon)
    r = policy_test(K)
    if (r > best_reward):
      best_reward = r
      best_K = K

  return best_K

def learn_controller (A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, 
    x_min=None, x_max=None, continuous=False, timestep=.01, rewardf=None, explore_mag=.04, step_size=.05, 
    coffset=None, bias=False, unsafe_flag=False, lqr_start=False):
    K = []
    if (learning_method == "lqr"):
        K = dlqr(A,B,Q,R)
        #K = lqr_gain(A,B,Q,R)
        print "K = {}".format(K)
        #print "double c[] = {%f, %f, %f, %f};" % (K[0,0], K[0,1], K[0,2], K[0,3])
    elif (learning_method == "random_search"):
        K = random_search_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,rewardf,explore_mag,step_size,coffset=coffset,bias=bias,unsafe_flag=unsafe_flag,lqr_start=lqr_start)
        print "K = {}".format(K)
    elif (learning_method == "random_search_2"):
        K = uniform_random_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep)
        print "K = {}".format(K)
    elif (learning_method == "policy_gradient"):
        K = policy_gradient_adam_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,rewardf,explore_mag,step_size,coffset=coffset)
        print "K = {}".format(K)
    else:
        print "Learning method {} is not found".format(learning_method)
    return K

def saveK (filename, K):
    np.save (filename, K)

def loadK (filename):
    return np.load (filename)

def draw_controller (A, B, K, x0, simulation_steps, names, continuous=False, timestep=.01, rewardf=None, coordination=None, coffset=None, bias=False):
  def f (x, u):
    return A.dot(x)+B.dot(u)

  return draw_controller_helper (f, K, x0, simulation_steps, names, continuous, timestep, rewardf, coordination, coffset, bias)    

def draw_controller_helper (f, K, x0, simulation_steps, names, continuous=False, timestep=.01, rewardf=None, coordination=None, coffset=None, bias=False):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")

    XS = []
    for i in range(len(names)):
        XS.append([])
    reward = 0
    for t in time:
        uk = K.dot(np.vstack([xk,[1]])) if bias else K.dot(xk)
        for i, k in enumerate(sorted(names.keys())):
            if coordination is None:
              val = xk[k,0]
              XS[i].append(val)
            else:
              val = xk[k,0]+coordination[k,0]
              XS[i].append(val)
        if rewardf is not None:
          reward += rewardf(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        if continuous:
          xk = xk + timestep*(f(xk, uk)) if coffset is None else xk + timestep*(f(xk, uk)+coffset)
        else:
          xk = f(xk, uk) if coffset is None else f(xk, uk)+coffset

    if rewardf is not None:
      print "Score of the trace: {}".format(reward) 

    for i, k in enumerate(sorted(names.keys())):
        plt.plot(time, XS[i], label=names[k])

    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    return xk

def test_controller (A, B, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    def f (x, u):
      return A.dot(x)+B.dot(u) 

    return test_controller_helper(f, K, x0, simulation_steps, rewardf, continuous, timestep, coffset, bias)

def test_controller_helper (f, K, x0, simulation_steps, rewardf, continuous=False, timestep=.01, coffset=None, bias=False):
    time = np.linspace(0, simulation_steps, simulation_steps, endpoint=True)
    xk = x0 #np.matrix(".0 ; 0 ; .0 ; 0.1")
    reward = 0
    for t in time:
        uk = K.dot(np.vstack([xk,[1]])) if bias else K.dot(xk)
        reward += rewardf(xk, uk)
        # Use discrete or continuous semantics based on user's choice
        if continuous:
          xk = xk + timestep*(f(xk, uk)) if coffset is None else xk + timestep*(f(xk, uk)+coffset)
        else:
          xk = f(xk, uk) if coffset is None else f(xk, uk)+coffset
    #print "Score of the trace: {}".format(reward) 
    return reward

def verify_controller (A, B, K, x_min, x_max, u_min, u_max, dimensions=[0,1]):
  """
  x_min = np.array([[-1.],[-1.]])
  x_max = np.array([[ 1.],[ 1.]])
  u_min = np.array([[-15.]])
  u_max = np.array([[ 15.]])
  """
  S = LinearSystem(A, B)
  X = Polyhedron.from_bounds(x_min, x_max)
  U = Polyhedron.from_bounds(u_min, u_max)
  D = X.cartesian_product(U)

  start = time.time()
  O_inf = S.mcais(K, D)
  end = time.time()
  print ("mcais execution time: {} secs".format(end - start))

  #if (len(dimensions) >= 2):
  #  D.plot(dimensions, label=r'$D$', facecolor='b')
  #  O_inf.plot(dimensions, label=r'$\mathcal{O}_{\infty}$', facecolor='r')
  #  plt.legend()
  #  plt.show()
  return O_inf

def verify_controller_via_discretization(Acl, h, x_min, x_max):
  #discretize the system for efficient verification
  X = Polyhedron.from_bounds(x_min, x_max)
  O_inf = mcais(la.expm(Acl * h), X, verbose=False)
  # dimensions=[0,2]
  # X.plot(dimensions, label=r'$D$', facecolor='b')
  # O_inf.plot(dimensions, label=r'$\mathcal{O}_{\infty}$', facecolor='r')
  # plt.legend()
  # plt.show()
  return O_inf

def dxdt(A, coffset=None): 
  # Constructing the vector field dx/dt = f
  #f = [-x[1]^3-x[1]*x[3]^2,
  #     -x[2]-x[1]^2*x[2],
  #-x[3]+3*x[1]^2*x[3]-3*x[3]/(x[3]^2+1)]
  d, p = A.shape
  X = []
  for i in range(p):
    X.append("x[" + str(i+1) + "]")

  f = []
  for i in range(len(A)):
    strstr = ""
    for k in range(len(X)):
      if (strstr is ""):
        strstr = str(A[i,k]) + "*" + X[k]
      else:
        strstr = strstr + "+" + str(A[i,k]) + "*" + X[k]
    if coffset is not None:
      strstr += ("+" + str(coffset[i,0]))
    f.append(strstr)
  return f  

def K_to_str (K):
  #Control policy K to text 
  nvars = len(K[0])
  X = []
  for i in range(nvars):
    X.append("x[" + str(i+1) + "]")

  ks = []
  for i in range(len(K)):
    strstr = ""
    for k in range(len(X)):
      if (strstr is ""):
        strstr = str(K[i,k]) + "*" + X[k]
      else:
        strstr = strstr + "+" + str(K[i,k]) + "*" + X[k]
    ks.append(strstr)
  return ks

def writeSOS(fname, sostext):
  file = open(fname,"w") 
  file.write(sostext) 
  file.close()
  return fname

def get_julia_path():
  if platform.system() == "Linux":
    return "julia"
  else:
    return "/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia"

def verifySOS(sosfile, quite, timeout, aggressive=False):
  def logged_sys_call(args, quiet, timeout):
    print "exec: " + " ".join(args)
    if quiet:
      out = open("result.log", "a")
    else:
      out = None
    kill = lambda process: process.kill()  
    julia = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout, kill, [julia])

    bcresult = None
    try:
      timer.start()
      bcresult = julia.communicate()
      if (aggressive):
        if (bcresult[0].find("Solution status : OPTIMAL") >= 0 and bcresult[1].split("#")[0] != "Optimal"):
          bcresult = "Optimal" + "#" + bcresult[1].split("#")[1]
        else:
          bcresult = bcresult[1]
      else:
        bcresult = bcresult[1]
    finally:
      timer.cancel()
      poll = julia.poll()
      if poll < 0:
        print("------------ Time-outs! ------------ ")
        os.system("killall -9 julia");
        child = subprocess.Popen(["pgrep julia"], stdout=subprocess.PIPE, shell=True)
        while True:
          result = child.communicate()[0]
          if result == "":
            break
    return bcresult
  #call /Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia ./sos.jl
  juliapath = get_julia_path()
  return logged_sys_call([juliapath] + [("%s" % sosfile)], quite, timeout)

def synthesize_verifed_controller(x0, A, B, Q, R, 
                      eq_err, learning_method, 
                      number_of_rollouts, simulation_steps, verification_steps,
                      s_min, s_max, x_min=None, x_max=None, 
                      avoid_list=None, avoid_list_dynamic=None,
                      continuous=False, timestep=.01, rewardf=None, 
                      explore_mag=.04, step_size=.05, coffset=None, 
                      K=None):
  
  safe = (x_min, x_max)
  target = (x_min, x_max)
  Theta = (s_min, s_max)

  def verification_oracle(x, initial_size, Theta, K):
    return bounded_z3(x, initial_size, Theta, K, A, B, target, safe, avoid_list, avoid_list_dynamic, verification_steps)

  def learning_oracle(x):
    if K is not None:
      return K
    else:
      return learn_controller (A, B, Q, R, x, eq_err, learning_method, number_of_rollouts, simulation_steps, 
                x_min, x_max, continuous, timestep, rewardf, explore_mag, step_size, coffset)

  return verify_controller_z3(x0, Theta, verification_oracle, learning_oracle)

@timeit
def learn_shield(A, B, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, actor, x_min, x_max, 
    rewardf=None, continuous=False, timestep=.005, explore_mag=.04, step_size=.05, 
    coffset=None, bias=False, unsafe_flag=False, lqr_start=False, without_nn_guide=False):

    def reward_func(x, Q, u, R):
      """
        the smaller the distance between the ouput of NN and linear controller,
        the higher reward.
        distance is measured by L1 distance, np.abs(actor.predict(x) - u) 
        u, Q, and R are useless here, reserved for the interface design.
      """
      sim_score = 0 if actor is None else -np.matrix([[np.sum(np.abs(actor.predict(np.reshape(x, (-1, actor.s_dim))) - u))]])
      safe_score = 0 if actor is not None or rewardf is None else rewardf(x, Q, u, R)
      return sim_score + safe_score

    if actor is None and rewardf is None:
      shield_reward = None
    elif not without_nn_guide:
      shield_reward = reward_func
    else:
      shield_reward = rewardf

    if (learning_method == "random_search"):
        K = random_search_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,shield_reward,explore_mag,step_size,coffset=coffset,bias=bias,unsafe_flag=unsafe_flag,lqr_start=lqr_start)
        print "K = {}".format(K)
    elif (learning_method == "random_search_2"):
        K = uniform_random_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,shield_reward)
        print "K = {}".format(K)
    elif (learning_method == "policy_gradient"):
        K = policy_gradient_adam_linear_policy(A,B,Q,R,x0,eq_err,number_of_rollouts,simulation_steps,x_min,x_max,continuous,timestep,shield_reward,explore_mag,step_size,coffset=coffset)
        print "K = {}".format(K)
    else:
        print "Learning method {} is not found".format(learning_method)
    return np.matrix(K)

@timeit
def learn_polysys_shield(f, ds, us, Q, R, x0, eq_err, learning_method, number_of_rollouts, simulation_steps, actor, 
    rewardf=None, continuous=False, timestep=.005, explore_mag=.04, step_size=.05, coffset=None, bias=False, unsafe_flag=False, without_nn_guide=False):

    def reward_func(x, Q, u, R):
      """
        the smaller the distance between the ouput of NN and linear controller,
        the higher reward.
        distance is measured by L1 distance, np.abs(actor.predict(x) - u) 
        u, Q, and R are useless here, reserved for the interface design.
      """
      sim_score = 0 if actor is None else -np.matrix([[np.sum(np.abs(actor.predict(np.reshape(x, (-1, actor.s_dim))) - u))]])
      safe_score = 0 if actor is not None or rewardf is None else rewardf(x, Q, u, R)
      return sim_score + safe_score

    if actor is None and rewardf is None:
      shield_reward = None
    elif not without_nn_guide:
      shield_reward = reward_func
    else:
      shield_reward = rewardf

    K = random_search_helper (f, ds, us, Q, R, x0, eq_err, number_of_rollouts, simulation_steps, continuous=continuous, timestep=timestep, rewardf=shield_reward, explore_mag=explore_mag, step_size=step_size, coffset=coffset, bias=bias, unsafe_flag=unsafe_flag)  

    print "K = {}".format(K)
    return K

shield_testing_on_x_ep_len = 10
