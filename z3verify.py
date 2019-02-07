from z3 import *
import itertools
import numpy as np

########################
## Util functions
########################

def get_matrix_multiplication_constraint(A, x, y, m, n):
	# A is a m\times n matrix. Can be a constant or a variable matrix
	# x is n\times 1 a vector
	# y is a m\times 1 vector, so that Ax = y
	list_of_constraints = []
	for i in range(m):
		constraint = Sum([A[i][j]*x[j] for j in range(n)]) == y[i]
		list_of_constraints.append(constraint)
	return list_of_constraints

def get_point_polytope_membership_constraint(x, poly):
	#Dimension of x is n\times 1
	#Dimension of A is m\times n
	#Dimension of b is m\times 1
	#Returns constraints representing A\times x + b <= 0

	A = poly[0]
	b = poly[1]
	m = len(A)
	n = len(x)

	list_of_constraints = []
	for i in range(m):
		constraint = Sum([A[i][j]*x[j] for j in range(n)]) + b[i] <= 0
		list_of_constraints.append(constraint)
	return And(list_of_constraints)

def get_point_rectangle_membership_constraint(x, rect):
	#Dimension of x is n\times 1
	#Dimension of A is m\times n
	#Dimension of b is m\times 1
	#Returns constraints representing A\times x + b <= 0
	list_of_constraints = []
	i = 0
	for (lo, hi) in rect:
		if not (hi is None):
			list_of_constraints.append(x[i] <= hi)
		if not (lo is None):
			list_of_constraints.append(x[i] >= lo)
		i = i+1
	return And(list_of_constraints)

def get_point_membership_constraint(x, rect_or_poly):
	#asserts that x is inside rect_or_poly
	(flag, desc) = rect_or_poly
	if flag:
		return get_point_rectangle_membership_constraint(x, desc)
	else:
		return get_point_polytope_membership_constraint(x, desc)

def get_disjointness_constraint(x_dim, P1, P2, str_identifier):
	#P1 is a tuple (A, b), A is m1\times x_dim representing polytope Ax + b <= 0
	#P2 is a tuple (C, d), C is m2\times x_dim representing polytope Cx + d <= 0
	#x_dim is the number of columns in A and C each
	#We have to assert (E = [A, C], f = [b, d]) is empty
	#Using Farkas Lemma, we instead have to assert that there is a y for which E^T y = 0, f^T y > 0 (and not f^T y < 0, notice how we represent polytopes) and y >= 0
	#str_identifier is just a new identifier for the new y variables introduced
	A = P1[0]
	b = P1[1]
	C = P2[0]
	d = P2[1]
	m1 = len(b) 
	m2 = len(d)
	y_dim = m1 + m2

	y = [Real(str_identifier + "_%s" %(j+1)) for j in range(y_dim)]

	list_of_constraints = []
	#First assert E^T y = 0
	for i in range(x_dim):
		lst = [A[j][i]*y[j] for j in range(m1)] + [C[j][i]*y[j+m1] for j in range(m2)]
		list_of_constraints.append(Sum(lst) == 0)

	#Next assert f^T > 0
	lst = [b[j]*y[j] for j in range(m1)] + [d[j]*y[j+m1] for j in range(m2)]
	list_of_constraints.append(Sum(lst) > 0)

	#Last, assert that y >= 0
	for j in range(y_dim):
		list_of_constraints.append(y[j] >= 0)

	return And(list_of_constraints)

def get_disjointness_constraint_poly_incomplete(x_dim, cornerList_P1, P2):
	#cornerList_P1 is the list of corners of the 1st polytope. 
	#P2 = (A, b) is a polytope 
	#x_dim is the number entries in rect_1 and in rect_2
	#We have to assert that rect1 \cap P2 is empty

	A = P2[0]
	b = P2[1]
	m = len(b)

	list_of_constraints = []
	for i in range(m):
		l = []
		for v in cornerList_P1:
			l.append( Sum([A[i][j]*v[j] for j in range(x_dim)]) + b[i] > 0 )
		list_of_constraints.append(And(l))

	return Or(list_of_constraints)

def get_disjointness_constraint_rect(x_dim, rect_1, rect_2):
	#rect_1 = [(lo1, hi1), ...., (lok, hik)] is a rectangle 
	#rect_2 = [(lo1, hi1), ...., (lok, hik)] is a rectangle 
	#None stands for + or - infty as appropriate
	#x_dim is the number entries in rect_1 and in rect_2
	#We have to assert that rect1 \cap rect2 is empty

	list_of_constraints = []
	for i in range(x_dim):
		lo_i_1 = rect_1[i][0]
		hi_i_1 = rect_1[i][1]
		lo_i_2 = rect_2[i][0]
		hi_i_2 = rect_2[i][1]
		if(not(lo_i_2 is None)):
			list_of_constraints.append(hi_i_1 < lo_i_2)
		if(not(hi_i_2 is None)):
			list_of_constraints.append(lo_i_1 > hi_i_2)

	return Or(list_of_constraints)

def get_rect_poly_containment_constraint(rect, poly):
	#rect = [(lo_1, hi_1), (lo_2, hi_2), ..., (lo_k, hi_k)] represents a polytope, with k = dimension of the state space.
	#target_poly = (target_mat, target_vec) is the target polyhedron
	#We will check if each of the vertices of the reach set is inside the target polyhedron
	corner_list = set(list(itertools.product(*rect)))

	list_of_constraints = []
	for corner in corner_list:
		list_of_constraints.append(get_point_polytope_membership_constraint(corner, poly))
	return And(list_of_constraints)

def get_rect_rect_containment_constraint(rect1, rect2):
	#rect1 and rect2 = [(lo_1, hi_1), (lo_2, hi_2), ..., (lo_k, hi_k)] represents a polytope, with k = dimension of the state space.
	#We will check if in each of the dimensions, rect1 \subseteq rect2
	#rect1 is assumed to be bounded

	dim = len(rect1)

	list_of_constraints = []
	for i in range(dim):
		(lo1, hi1) = rect1[i]
		(lo2, hi2) = rect2[i]
		if not(lo2 is None):
			list_of_constraints.append(lo2 <= lo1)
		if not(hi2 is None):
			list_of_constraints.append(hi1 <= hi2)
	return And(list_of_constraints)

def get_rect_containment_constraint(rect, rect_or_poly):
	#asserts that rect is contained inside rect_or_poly
	(flag, desc) = rect_or_poly
	if flag:
		return get_rect_rect_containment_constraint(rect, desc)
	else:
		return get_rect_poly_containment_constraint(rect, desc)

def get_next_constraint_point(x, A, B, K, x_dim, next_x):
	list_of_constraints = []
	A_cl = A + B.dot(K)
	for i in range(x_dim):
		constraint = Sum([A_cl[i,j]*x[j] for j in range(x_dim)]) == next_x[i]
		list_of_constraints.append(constraint)
	return And(list_of_constraints)


########################
## Set functions
########################

option_initial = True
option_center = True

def get_avoid_constraint_rect(interval_list, avoid_list):
	#interval_list = [(lo_1, hi_1), (lo_2, hi_2), ..., (lo_k, hi_k)] represents a polytope, with k = dimension of the state space.
	dim = len(interval_list)
	corner_list = set(list(itertools.product(*interval_list)))

	list_of_constraints = []
	for avoid in avoid_list:
		(avoid_flag, avoid_desc) = avoid
		if avoid_flag:
			list_of_constraints.append(get_disjointness_constraint_rect(dim, interval_list, avoid_desc))
		else:
			list_of_constraints.append(get_disjointness_constraint_poly_incomplete(dim, corner_list, avoid_desc))
	return And(list_of_constraints)

def get_avoid_constraint(x, avoid_list):
	list_of_constraints = []
	for avoid in avoid_list:
		list_of_constraints.append(get_point_membership_constraint(x, avoid))
	return And(list_of_constraints)

def add_constraints(s, x, Theta, x_dim, A, B, K, target, avoid_list, avoid_list_dynamic, num_steps):
	#s is a z3 Solver
	#x, u and r are z3 variables
	#Theta, is the inital set
	#radius_list = [lst_1, lst_2, ..., lst_m], where m= num_steps, and each lst_i = [r1, ..., r_k], k = x_dim is the per-dimension constant factor to be multiplied at the i'th step
	#A is a constant square matrix of dimension x_dim \times x_dim
	#u_dim is the dimension of the input space
	#B is the feedback matrix of dimension x_dim\times u_dim
	#u_poly = (u_mat, u_vec) represents the space of inputs. That is, we allow any u that satisfies u_mat\times u + u_vec <= 0
	#target_poly = (target_mat, target_vec) is the target polytope
	#avoid_list is a list [(flag1, poly1), (flag2, poly2) ...  (flagk, polyk)], where flagi = True means rectangle, o/w general polytope. if flagi = True, thn polyi = list of x_dim pairs (hi, low). O/w it is (Ai, bi).

	is_dynamic_list_None = (avoid_list_dynamic is None)

	s.push()
	s.add(get_avoid_constraint(x[0], avoid_list if is_dynamic_list_None else avoid_list + avoid_list_dynamic[0]))
	res = s.check()
	if (res == sat):
		#z3 said the property is violated
		ce = get_counterexample(s, x, i)
		s.pop()
		return ce
	elif (res == unsat):
		s.pop()
	else:
		print "Warning: Z3 returns unknown!"
		s.pop()	

	for i in range(num_steps):
		#u_i_constraint = get_point_membership_constraint(u[i], u_space)
		next_x_constraint = get_next_constraint_point(x[i], A, B, K, x_dim, x[i+1])
		safety_constraint = get_avoid_constraint(x[i+1], avoid_list if is_dynamic_list_None else avoid_list + avoid_list_dynamic[i+1])

		#s.add(u_i_constraint)
		s.add(next_x_constraint)

		s.push()
		s.add(safety_constraint)
		res = s.check()
		if (res == sat):
			#z3 said the property is violated
			ce = get_counterexample(s, x, i)
			s.pop()
			return ce
		elif (res == unsat):
			s.pop()
		else:
			print "Warning: Z3 returns unknown!"
			s.pop()


	reach_constraint = get_point_membership_constraint(x[num_steps], target)
	s.add(Not(reach_constraint))
	res = s.check()
	if (res == sat):
		#z3 said the property is violated
		return get_counterexample(s, x, num_steps)
	elif (res == unsat):
		return None
	else:
		print "Warning: Z3 returns unknown!"
		return None 

def add_constraints_safety(s, x, Theta, x_dim, A, B, K, target, safe, num_steps):
	#s is a z3 Solver
	#x, u and r are z3 variables
	#Theta, is the inital set
	#radius_list = [lst_1, lst_2, ..., lst_m], where m= num_steps, and each lst_i = [r1, ..., r_k], k = x_dim is the per-dimension constant factor to be multiplied at the i'th step
	#A is a constant square matrix of dimension x_dim \times x_dim
	#u_dim is the dimension of the input space
	#B is the feedback matrix of dimension x_dim\times u_dim
	#u_poly = (u_mat, u_vec) represents the space of inputs. That is, we allow any u that satisfies u_mat\times u + u_vec <= 0
	#target_poly = (target_mat, target_vec) is the target polytope
	#safe is the safety set (invariant for the system)

	s.push()
	s.add(Not(get_point_membership_constraint(x[0], safe)))
	res = s.check()
	if (res == sat):
		#z3 said the property is violated
		ce = get_counterexample(s, x, i)
		s.pop()
		return ce
	elif (res == unsat):
		s.pop()
	else:
		print "Warning: Z3 returns unknown!"
		s.pop()

	for i in range(num_steps):
		#u_i_constraint = get_point_membership_constraint(u[i], u_space)
		next_x_constraint = get_next_constraint_point(x[i], A, B, K, x_dim, x[i+1])
		safety_constraint = get_point_membership_constraint(x[i+1], safe)

		#s.add(u_i_constraint)
		s.add(next_x_constraint)

		s.push()
		s.add(Not(safety_constraint))
		res = s.check()
		if (res == sat):
			#z3 said the property is violated
			ce = get_counterexample(s, x, i)
			s.pop()
			return ce
		elif (res == unsat):
			s.pop()
		else:
			print "Warning: Z3 returns unknown!"
			s.pop()

	reach_constraint = get_point_membership_constraint(x[num_steps], target)
	s.add(Not(reach_constraint)) 
	res = s.check()
	if (res == sat):
		#z3 said the property is violated
		return get_counterexample(s, x, num_steps)
	elif (res == unsat):
		return None
	else:
		print "Warning: Z3 returns unknown!"
		return None

def get_counterexample(s, x, num_steps):
	# A counterexample is found
	m = s.model()
	# print("model=", m)
	trajectory = [[ (m[x[i][j]].numerator_as_long())*1.0/(m[x[i][j]].denominator_as_long()) for j in range(x_dim)] for i in range(num_steps+1)]
	
	return trajectory[0]


########################
## Verify Controllers using Z3
########################

multiplicative_factor_for_radius = 0.95
max_decrease_steps = 200
max_num_iters = 10000
print_detail = False
# option_center = True

def check_covered(dim, set1, list_of_covers, useInv):
	x = [Real("x_[%s]" %(j+1)) for j in range(dim)]
	list_of_constraints = []

	#x \in set1
	list_of_constraints.append(get_point_membership_constraint(x, set1))

	#x is not in any of the covers
	if not (useInv):
		for cover in list_of_covers:
			list_of_constraints.append(Not(get_point_membership_constraint(x, cover)))

	s_cover = Solver()
	s_cover.add(And(list_of_constraints))

	#x is not in any of the covers
	if (useInv):
		for cover in list_of_covers:
			cover_cnstr = eval(cover)
			s_cover.add(Not(cover_cnstr))

	chk = s_cover.check()
	if(chk == sat):
		# A counterexample is found
		m = s_cover.model()
		# print("model=", m)
		return np.matrix([[ (m[x[j]].numerator_as_long())*1.0/(m[x[j]].denominator_as_long()) ] for j in range(dim)])
	elif(chk == unsat):
		return None
	else:
		#Some problem
		print("There has been some problem in checking if the initial set has been covered")
		return None

def d2tod1(space):
	(x_min, x_max) = space
	return (True, [ (x_min[i][0], x_max[i][0]) for i in range(len(x_min)) ])

def bounded_z3(x0, initial_size, Theta, K, A, B, target, safe, avoid_list, avoid_list_dynamic, num_steps):
	#2d array target, safe, Theta is simplied to 1d
	target = d2tod1(target)
	safe = d2tod1(safe)

	x_dim = len(x0)

	x = [[Real("x_ref_%s[%s]" %(i, j+1)) for j in range(x_dim)] for i in range(num_steps+1)]
	#u = [[Real("u_ref_%s[%s]" %(i, j+1)) for j in range(u_dim)] for i in range(num_steps)]

	# Constraits for resricting the verifier's attension only on a region near the input
	initial_rectangle = (True, [(x0[i,0]-initial_size[i], x0[i,0]+initial_size[i]) for i in range(x_dim)])

	print "initial_rectangle = {}".format(initial_rectangle)

	s = Solver()
	
	if print_detail:
		print("Adding intial and transition constraints ... ")

	s.add(get_point_membership_constraint(x[0], initial_rectangle))
	
	if print_detail:
		print("Now adding Theta constraints ... ")

	print "Theta = {}".format(Theta)
	s.add(get_point_membership_constraint(x[0], Theta))

	s_check = None
	if safe is None:
		s_check = add_constraints(s, x, Theta, x_dim, A, B, K, target, avoid_list, avoid_list_dynamic, num_steps)
	else:
		s_check = add_constraints_safety(s, x, Theta, x_dim, A, B, K, target, safe, num_steps)

	if (s_check is not None):
		if print_detail:
			print("Found a counterexample trajectory from:", trajectory[0])

		if print_detail:
			print("Fails for:", initial_size)
		return False
	else:
		return True

def verify_controller_z3(x0, Theta, verification_oracle, learning_oracle, draw_oracle, continuous):		
	if continuous:
		x_dim = len(x0)
		print "Dimension of the system: {}".format(x_dim)
		(s_min, s_max) = Theta
		#2d array target, safe, Theta is simplied to 1d
		Theta = d2tod1(Theta)

		Theta_has_been_covered = False
		number_of_steps_initial_size_has_been_halved = 0
		num_iters = 0
		covered_list = []

		if print_detail:
			print("Starting iterations")

		K = learning_oracle(x0)
		draw = True
		initial_size = np.array([max(s_max[i,0] - x0[i, 0], x0[i, 0]- s_min[i,0]) for i in range(x_dim)])

		l = 0
		h = initial_size
		width = h
		pvresult = None
		pvinv = None
		pinitial_size = None
		resultList = []

		# Too many variables can kill z3's performance.
		useInv = (x_dim <= 2) # Todo: more experiments to find a reasonable threshold.

		while((not Theta_has_been_covered) and number_of_steps_initial_size_has_been_halved < max_decrease_steps and num_iters < max_num_iters):
			num_iters = num_iters + 1
			print ("At {} iteration work on the input:\n {}".format(num_iters, x0))

			# Tell if a policy is worthwhile for verification
			while(draw):
				print "Learning algorithm finds a controller K = {}".format(K)
				if (draw_oracle is not None):
					test_result = draw_oracle (x0, K)
					print ("If the the following trjectory socre is {} >= 0, then safe!".format(test_result))  				
	 				if (test_result >= 0):
	 					print ("Learned controller looks Okay for {}".format(x0))
	 					break
	  				else:
	  					print ("Learning controller is not Okay for {}".format(x0))
	  					K = learning_oracle(x0)
	  					continue
	  			else:
	  				break

	  		print ("intial_size: {}".format(initial_size))
			vresult, vinv = verification_oracle(x0, initial_size, Theta, K)
			if (vresult):
				pvresult = vresult
				pvinv = vinv
				pinitial_size = initial_size


			#The K may do better than what is constrained by initial_size
			# l_vresult = vresult
			# l_initial_size = initial_size
			# while (l_vresult):
			# 	print ("Trying to increase the covered space of a verified controller")
			# 	vresult = l_vresult
			# 	initial_size = l_initial_size
			# 	l_initial_size = l_initial_size / multiplicative_factor_for_radius
			# 	stop_condition = True
			# 	for index in range(x_dim):
			# 		if (l_initial_size[index] < o_initial_size[index]):
			# 			stop_condition = False
			# 			break
			# 	if stop_condition:
			# 		break
			# 	l_vresult = verification_oracle(x0, l_initial_size, Theta, K)

			print "Verification algorithm finds the controller {}".format(vresult)
			if vresult:
				print "Verification algorithm finds the inducive invariant for the controller {}".format(vinv)

			if (number_of_steps_initial_size_has_been_halved == 0 and vresult):
				resultList.append((x0, pinitial_size, pvinv, K))
				return True, resultList
			elif (not (h - l < width * (1 - multiplicative_factor_for_radius)).any()) and (not useInv or not vresult):
				# A counterexample is found
				#initial_size = initial_size * multiplicative_factor_for_radius
				if (number_of_steps_initial_size_has_been_halved == 0):
					initial_size = (l + h) / 2
				elif (vresult):
					l = (l + h) / 2
					initial_size = (l + h) / 2
				else:
					h = (l + h) / 2
					initial_size = (l + h) / 2
				number_of_steps_initial_size_has_been_halved = number_of_steps_initial_size_has_been_halved + 1
				draw = False # Work again on the sample counterexample so there is no need to redraw a picture
			else:
				if (pvresult is None or pvinv is None or pinitial_size is None):
					# K seems not good to the verifier; restart a new search for K.
					number_of_steps_initial_size_has_been_halved = 0
					K = learning_oracle(x0)
					initial_size = np.array([max(s_max[i,0] - x0[i, 0], x0[i, 0]- s_min[i,0]) for i in range(x_dim)])
					l = 0
					h = initial_size
					width = h
					pvresult = None
					pvinv = None
					pinitial_size = None
					draw = True
					#return False, resultList
				else: 
					resultList.append((x0, pinitial_size, pvinv, K))

					if not (useInv):
						cover = []
						if type(initial_size) is np.ndarray:
							for i in range(len(x0)):
								x_i = x0[i,0]
								cover.append((x_i-initial_size[i], x_i+initial_size[i]))
						else:
							for i in range(len(x0)):
								x_i = x0[i,0]
								cover.append((x_i-initial_size, x_i+initial_size))
						covered_list.append((True,cover))
						x0 = check_covered(x_dim, Theta, covered_list, False) 
					else:
						vinv = barrier_certificate_str2z3(pvinv, x_dim)
						covered_list.append(vinv)
						x0 = check_covered(x_dim, Theta, covered_list, True) 


					Theta_has_been_covered = (x0 is None) or (number_of_steps_initial_size_has_been_halved == 0)
					number_of_steps_initial_size_has_been_halved = 0
					if (not Theta_has_been_covered):
						K = learning_oracle(x0)
						initial_size = np.array([max(s_max[i,0] - x0[i, 0], x0[i, 0]- s_min[i,0]) for i in range(x_dim)])
						l = 0
						h = initial_size
						width = h
						pvresult = None
						pvinv = None
						pinitial_size = None
						draw = True

		print("Number of iterations: " + str(num_iters) + "; Theta_has_been_covered: " + str(Theta_has_been_covered))
		return Theta_has_been_covered, resultList
	else:
		x_dim = len(x0)
		print "Dimension of the system: {}".format(x_dim)
		(s_min, s_max) = Theta
		#2d array target, safe, Theta is simplied to 1d
		Theta = d2tod1(Theta)

		Theta_has_been_covered = False
		number_of_steps_initial_size_has_been_halved = 0
		num_iters = 0
		covered_list = []

		if print_detail:
			print("Starting iterations")

		K = learning_oracle(x0)
		draw = True
		initial_size = np.array([max(s_max[i,0] - x0[i, 0], x0[i, 0]- s_min[i,0]) for i in range(x_dim)])

		while((not Theta_has_been_covered) and number_of_steps_initial_size_has_been_halved < max_decrease_steps and num_iters < max_num_iters):
			num_iters = num_iters + 1
			print ("At {} iteration work on the input:\n {}".format(num_iters, x0))

			# Tell if a policy is worthwhile for verification
			while(draw):
				print "Learning algorithm finds a controller K = {}".format(K)
				if (draw_oracle is not None):
					test_result = draw_oracle (x0, K)
					print ("If the the following trjectory socre is {} >= 0, then safe!".format(test_result))  				
	 				if (test_result >= 0):
	 					print ("Learned controller looks Okay for {}".format(x0))
	 					break
	  				else:
	  					print ("Learning controller is not Okay for {}".format(x0))
	  					K = learning_oracle(x0)
	  					continue
	  			else:
	  				break

	  		print ("intial_size: {}".format(initial_size))
			vresult = verification_oracle(x0, initial_size, Theta, K)

			#The K may do better than what is constrained by initial_size
			# l_vresult = vresult
			# l_initial_size = initial_size
			# while (l_vresult):
			# 	print ("Trying to increase the covered space of a verified controller")
			# 	vresult = l_vresult
			# 	initial_size = l_initial_size
			# 	l_initial_size = l_initial_size / multiplicative_factor_for_radius
			# 	stop_condition = True
			# 	for index in range(x_dim):
			# 		if (l_initial_size[index] < o_initial_size[index]):
			# 			stop_condition = False
			# 			break
			# 	if stop_condition:
			# 		break
			# 	l_vresult = verification_oracle(x0, l_initial_size, Theta, K)

			print "Verification algorithm finds the controller {}".format(vresult)

			if (not vresult):
				# A counterexample is found
				initial_size = initial_size * multiplicative_factor_for_radius
				number_of_steps_initial_size_has_been_halved = number_of_steps_initial_size_has_been_halved + 1
				draw = False # Work again on the sample counterexample so there is no need to redraw a picture
			else:
				cover = []
				if type(initial_size) is np.ndarray:
					for i in range(len(x0)):
						x_i = x0[i,0]
						cover.append((x_i-initial_size[i], x_i+initial_size[i]))
				else:
					for i in range(len(x0)):
						x_i = x0[i,0]
						cover.append((x_i-initial_size, x_i+initial_size))
				covered_list.append((True,cover))
				x0 = check_covered(x_dim, Theta, covered_list, False) 
				Theta_has_been_covered = (x0 is None) or (number_of_steps_initial_size_has_been_halved == 0)
				number_of_steps_initial_size_has_been_halved = 0
				if (not Theta_has_been_covered):
					K = learning_oracle(x0)
					initial_size = np.array([max(s_max[i,0] - x0[i, 0], x0[i, 0]- s_min[i,0]) for i in range(x_dim)])
					draw = True

		print("Number of iterations: " + str(num_iters) + "; Theta_has_been_covered: " + str(Theta_has_been_covered))
		return Theta_has_been_covered



import re
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