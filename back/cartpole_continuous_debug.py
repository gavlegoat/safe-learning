# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2018-10-23 22:14:33
# Last Modified by:   Zikang Xiong
# Last Modified time: 2018-10-23 22:24:54
# -------------------------------
from main import *
import sys
from DDPG import *

def cartpole (learning_method, number_of_rollouts, simulation_steps, K=None):
  A = np.matrix([
  [0, 1,     0, 0],
  [0, 0, 0.716, 0],
  [0, 0,     0, 1],
  [0, 0, 15.76, 0]
  ])
  B = np.matrix([
  [0],
  [0.9755],
  [0],
  [1.46]
  ])

   #intial state space
  s_min = np.array([[ -0.1],[ -0.1], [-0.05], [ -0.05]])
  s_max = np.array([[  0.1],[  0.1], [ 0.05], [  0.05]])

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

  args = { 'actor_lr': 0.0001,
         'critic_lr': 0.001,
         'actor_structure': actor_structure,
         'critic_structure': critic_structure, 
         'buffer_size': 1000000,
         'gamma': 0.99,
         'max_episode_len': 500,
         'max_episodes': learning_eposides,
         'minibatch_size': 64,
         'random_seed': 6553,
         'tau': 0.005,
         'model_path': train_dir+"model.chkp",
         'enable_test': False, 
         'test_episodes': 1,
         'test_episodes_len': 5000}

  actor, env = DDPG(A, B, u_min, u_max, s_min, s_max, x_min, x_max, Q, R, args=args, continuous=True)
  
  #################### Train Shield #################
  K = np.array([[  5.40839089,  12.07876949, -61.3628737,  -18.08569268]])

  B1_str = r"138.57196079025456x1^4 + 132.57688268997495x1^3x2 - 715.8453327705934x1^3x3 - 96.61903326765665x1^3x4 + 188.61676048838999x1^2x2^2 - 858.996856971179x1^2x2x3 - 297.43141466965653x1^2x2x4 + 2217.587686760044x1^2x3^2 + 689.8260157201024x1^2x3x4 + 128.14599768506284x1^2x4^2 + 3.102160655380709x1x2^3 - 203.48088085117234x1x2^2x3 - 10.060321775241173x1x2^2x4 + 705.8228304767141x1x2x3^2 + 335.6571505741948x1x2x3x4 + 4.282821121537309x1x2x4^2 - 1125.0433149080256x1x3^3 - 547.2095597002182x1x3^2x4 - 127.7033706042299x1x3x4^2 - 2.5293351881503257x1x4^3 + 81.16935093976316x2^4 - 363.7937617876956x2^3x3 - 236.72521262116376x2^3x4 + 1074.7216635601744x2^2x3^2 + 835.5794991330489x2^2x3x4 + 261.71266061954543x2^2x4^2 - 1253.8153727594677x2x3^3 - 1690.9628616343878x2x3^2x4 - 619.5922405076799x2x3x4^2 - 133.86272262189866x2x4^3 + 2323.8984705619027x3^4 + 1015.3884215929708x3^3x4 + 709.8225461803003x3^2x4^2 + 149.43006198516312x3x4^3 + 30.333852929304562x4^4 + 4.9504240585838114e-15x1^3 + 3.2059630978282643e-15x1^2x2 - 5.748882750204367e-15x1^2x3 - 2.214496594600989e-15x1^2x4 + 7.208807200940311e-15x1x2^2 - 1.943375137461502e-14x1x2x3 - 1.0866178308242248e-14x1x2x4 - 4.0347787971662873e-14x1x3^2 + 1.5695326575564025e-14x1x3x4 + 2.7739752389799254e-15x1x4^2 - 9.988851036507937e-15x2^3 + 1.3843750318224787e-15x2^2x3 + 2.203290878450811e-14x2^2x4 - 2.9365998660948163e-14x2x3^2 - 2.3205383055557544e-15x2x3x4 - 1.628001492828702e-14x2x4^2 + 1.776520841657554e-14x3^3 + 2.3564989300149288e-14x3^2x4 + 1.2609073869631834e-15x3x4^2 + 4.133253598603063e-15x4^3 + 532.9878617065255x1^2 + 234.55084094205833x1x2 - 1049.731220698885x1x3 - 181.34707891023734x1x4 + 246.74380841369367x2^2 - 834.9586170244658x2x3 - 374.5490720117498x2x4 + 1859.8032592942388x3^2 + 679.7644799667256x3x4 + 152.5808904420382x4^2 - 31.635042356706144"
  B2_str = r"5637.545200332092x1^4 + 6519.337364869608x1^3x2 - 39540.827423175404x1^3x3 - 4835.052957746221x1^3x4 + 3036.2784249062697x1^2x2^2 - 26912.062668054983x1^2x2x3 - 6459.818647051525x1^2x2x4 + 100085.83424587506x1^2x3^2 + 23627.79992955291x1^2x3x4 + 3312.5680014217273x1^2x4^2 - 3115.651315770129x1x2^3 + 22640.373053388386x1x2^2x3 + 6290.838902487829x1x2^2x4 - 56231.15269790099x1x2x3^2 - 35607.76908421783x1x2x3x4 - 6041.033623793844x1x2x4^2 + 50420.46482509308x1x3^3 + 51587.6854106116x1x3^2x4 + 15483.263126066035x1x3x4^2 + 950.2888113202259x1x4^3 + 9552.744508426587x2^4 - 51597.92656658118x2^3x3 - 26923.193281327014x2^3x4 + 233189.18114160903x2^2x3^2 + 117778.5485485714x2^2x3x4 + 29608.61733301012x2^2x4^2 - 450802.5362611953x2x3^3 - 354023.07067891466x2x3^2x4 - 85480.11588172935x2x3x4^2 - 14925.326631799535x2x4^3 + 575453.0035540356x3^4 + 363090.8238027895x3^3x4 + 148198.6754928286x3^2x4^2 + 22991.528612496146x3x4^3 + 4120.681466829561x4^4 - 3.8128254560981964e-14x1^3 - 1.1059005283823343e-13x1^2x2 + 4.4797530643260733e-13x1^2x3 + 7.360575986189389e-14x1^2x4 - 7.85666145116758e-14x1x2^2 + 2.125509312218588e-13x1x2x3 + 1.172567507019689e-13x1x2x4 - 2.242065634766375e-12x1x3^2 - 1.6387849787595076e-13x1x3x4 - 6.138349520631801e-14x1x4^2 - 4.7124066571889376e-14x2^3 + 4.9289146227871674e-14x2^2x3 + 8.899640969527001e-14x2^2x4 + 7.221068678383892e-13x2x3^2 - 1.4578339687495474e-13x2x3x4 - 2.7305881111134045e-14x2x4^2 - 1.1922478695515645e-12x3^3 - 5.671728058376497e-13x3^2x4 + 6.315790656152508e-14x3x4^2 - 4.411687123782701e-15x4^3 + 300.75005900180406x1^2 + 522.9710565510807x1x2 - 856.2849827559211x1x3 - 388.7459672510427x1x4 + 1067.3562321912377x2^2 - 1821.4523755988498x2x3 - 1591.7439484339368x2x4 + 7056.533495704923x3^2 + 1730.467477231916x3x4 + 669.9081616935501x4^2 - 104.0258086100814"
  B3_str = r"3.625225930877096x1^4 + 4.131766754174828x1^3x2 - 20.024007902670736x1^3x3 - 3.0321679643794406x1^3x4 + 5.146456767185643x1^2x2^2 - 24.67793288759264x1^2x2x3 - 8.097455302802828x1^2x2x4 + 62.689150309452174x1^2x3^2 + 19.898769962324813x1^2x3x4 + 3.4778704197631654x1^2x4^2 + 0.3854869811271967x1x2^3 - 6.747973765412874x1x2^2x3 - 0.913111251322851x1x2^2x4 + 17.501459070619035x1x2x3^2 + 11.018823457173001x1x2x3x4 + 0.5440470888094361x1x2x4^2 - 26.228107088315372x1x3^3 - 13.495426404791393x1x3^2x4 - 4.026641277009685x1x3x4^2 - 0.21040798280068976x1x4^3 + 2.755132511061368x2^4 - 11.866450790405782x2^3x3 - 8.056312454572966x2^3x4 + 37.18770994801018x2^2x3^2 + 27.17875929548609x2^2x3x4 + 8.97642036062069x2^2x4^2 - 35.26480142082856x2x3^3 - 58.508407591863886x2x3^2x4 - 19.77791412095364x2x3x4^2 - 4.689688530845067x2x4^3 + 92.81415908664876x3^4 + 28.64907371045615x3^3x4 + 25.085826738723078x3^2x4^2 + 4.5482598553411275x3x4^3 + 1.117589569344238x4^4 + 2.783885563892249e-16x1^3 - 2.1678302190037319e-16x1^2x2 - 1.165974511140296e-15x1^2x3 + 1.5718276712286477e-16x1^2x4 + 3.6759959156370747e-16x1x2^2 + 6.497883529680794e-16x1x2x3 - 5.90981903151833e-16x1x2x4 + 3.216747143259401e-15x1x3^2 - 5.365525323788355e-16x1x3x4 + 2.8849292734417527e-16x1x4^2 + 1.4025989665327955e-16x2^3 - 9.971655493644465e-16x2^2x3 - 2.659106156471031e-16x2^2x4 + 1.0370995318102337e-15x2x3^2 + 1.5980338023408934e-15x2x3x4 + 1.7225623117402264e-16x2x4^2 + 2.2020299913355668e-15x3^3 - 8.372952730439513e-16x3^2x4 - 5.636122199933437e-16x3x4^2 - 5.5320488696050545e-17x4^3 + 6.99680808794137x1^2 + 6.123367728233026x1x2 - 18.957933725968385x1x3 - 4.601328661911071x1x4 + 7.592201316602356x2^2 - 19.048289741338884x2x3 - 11.556721785096292x2x4 + 76.90790807606801x3^2 + 15.667704139971725x3x4 + 5.0161053749811915x4^2 - 2.1281257143880223"
  B4_str = r"2448.2933916899506x1^4 + 2709.540294750388x1^3x2 - 16112.647941178011x1^3x3 - 2009.2619094217134x1^3x4 + 2659.468631625977x1^2x2^2 - 14736.350849915974x1^2x2x3 - 4581.96201261926x1^2x2x4 + 47391.91728765927x1^2x3^2 + 12305.671436228666x1^2x3x4 + 2109.0601850075777x1^2x4^2 - 33.176788113437276x1x2^3 - 523.7344254092902x1x2^2x3 - 133.9042907331295x1x2^2x4 - 39.264746367526726x1x2x3^2 + 1020.8414859006555x1x2x3x4 - 468.1262081130273x1x2x4^2 - 28862.961541774985x1x3^3 + 1697.3581265383743x1x3^2x4 - 227.7856570784384x1x3x4^2 + 20.583014664610417x1x4^3 + 3189.867981680884x2^4 - 15372.549013137505x2^3x3 - 9453.52498234076x2^3x4 + 59441.48102956826x2^2x3^2 + 35553.522381213385x2^2x3x4 + 10281.739735378178x2^2x4^2 - 112263.60494715226x2x3^3 - 90880.88474347886x2x3^2x4 - 25072.614875308293x2x3x4^2 - 5198.117153280107x2x4^3 + 236426.635068644x3^4 + 90567.31269373423x3^3x4 + 40637.65986696805x3^2x4^2 + 6475.572301322802x3x4^3 + 1702.5442124797175x4^4 - 3.222706470300374e-14x1^3 - 2.1514849603649008e-14x1^2x2 + 1.9836064202962772e-13x1^2x3 - 1.2631364157982617e-14x1^2x4 + 1.0145697747106574e-13x1x2^2 + 7.226569458157116e-13x1x2x3 - 2.1124599319971196e-13x1x2x4 + 7.141255374318527e-14x1x3^2 - 5.998206279085732e-13x1x3x4 + 1.2127279651764744e-13x1x4^2 + 3.634296110031192e-14x2^3 - 2.7642161539314464e-13x2^2x3 - 1.6504259954479648e-14x2^2x4 - 1.0952970875354544e-12x2x3^2 + 5.13055799059482e-13x2x3x4 - 7.651855599853246e-14x2x4^2 + 7.392643432333319e-13x3^3 + 9.724027384356073e-13x3^2x4 - 2.360466497687471e-13x3x4^2 + 5.594427019686017e-14x4^3 + 278.72310001738964x1^2 + 717.1708414570119x1x2 - 2865.0068952925963x1x3 - 754.3249637181683x1x4 + 1109.4350867043818x2^2 - 4786.54844138508x2x3 - 2345.9941336497536x2x4 + 21195.9319573455x3^2 + 5148.204499837994x3x4 + 1785.5289888910497x4^2 - 161.0060152611757"
 
  B1 = barrier_certificate_str2func(B1_str, len(x_max))
  B2 = barrier_certificate_str2func(B2_str, len(x_max))
  B3 = barrier_certificate_str2func(B3_str, len(x_max))
  B4 = barrier_certificate_str2func(B4_str, len(x_max))

  ############ Use Shield ###############
  fail_time = 0
  success_time = 0
  fail_list = []
  shield_count = 0

  print ("Test Shield")

  for ep in xrange(args['test_episodes']):
    errorinput = np.matrix([[-0.07066096],[-0.06902439],[ 0.04788092],[-0.01430553]])
    x = env.resetX(errorinput)
    print ('sampled input {}'.format(x))
    init_x = x
    u = np.matrix(np.zeros(u_min.shape))
    xk = x

    test_init_state_in_shield = B1(*state2list(xk)) <= 0 and B2(*state2list(xk)) <= 0 and B3(*state2list(xk)) <= 0 and B4(*state2list(xk)) <= 0
    print ('init_state_in_shield: {}'.format(test_init_state_in_shield))

    # Tracke the value flow of B's along the trajectory
    preB1 = 10000
    preB2 = 10000
    preB3 = 10000
    preB4 = 10000

    lookaheadtimes = 0

    print "----ep: {} ----".format(ep)
    for i in xrange(args['test_episodes_len']):
      # simulation
      u = actor.predict(np.reshape(np.array(x), (1, actor.s_dim)))
      xk = env.simulation(u)

      
      #print("preB1 {}".format(preB1))
      #print("preB2 {}".format(preB2))
      #print("preB3 {}".format(preB3))
      #print("preB4 {}".format(preB4))
      #print("B1 {}".format(B1(*state2list(xk))))
      #print("B2 {}".format(B2(*state2list(xk))))
      #print("B3 {}".format(B3(*state2list(xk))))
      #print("B4 {}".format(B4(*state2list(xk))))

      currB1 = B1(*state2list(xk))
      currB2 = B2(*state2list(xk))
      currB3 = B3(*state2list(xk))
      currB4 = B4(*state2list(xk))

      safe = True

      # Fixeme: all following valuess like -50, 1, 5, 19 should be parameterized!
      # value of B's are increasing towards 0 or are taking very large increase!
      if (currB1 > -50 and int(currB1) - int(preB1) > 1) or (int(currB1) - int(preB1) > 5) or\
          (currB2 > -50 and int(currB2) - int(preB2) > 1) or (int(currB2) - int(preB2) > 5) or\
          (currB3 > -50 and int(currB3) - int(preB3) > 1) or (int(currB3) - int(preB3) > 5) or\
          (currB4 > -50 and int(currB4) - int(preB4) > 1) or (int(currB4) - int(preB4) > 5):
        #simulate sufficently long to check if B's will be above 0
        for k in range(20):
            uk = actor.predict(np.reshape(np.array(xk), (1, actor.s_dim)))
            xk = xk + 0.01 * (A.dot(xk) + B.dot(uk))
        currB1 = B1(*state2list(xk))
        currB2 = B2(*state2list(xk))
        currB3 = B3(*state2list(xk))
        currB4 = B4(*state2list(xk))
        lookaheadtimes = lookaheadtimes + 1

      #print ("currB1 {}".format(currB1))
      #print ("currB2 {}".format(currB2))
      #print ("currB3 {}".format(currB3))
      #print ("currB4 {}".format(currB4))
      
      # safe or not in next test steps
      if currB1 > 0 or currB2 > 0\
      or currB3 > 0 or currB4 > 0:  
        safe = False   
        
      if not safe:
        u = K.dot(x)
        print 'Shield! in state: \n', x
        print 'Shield! at step: \n', i
        shield_count += 1
      #else:
      #  print 'Not Shield! in state: \n', x

      # step
      x, _, terminal = env.step(u)
      
      # record value of B's
      preB1 = B1(*state2list(x))
      preB2 = B2(*state2list(x))
      preB3 = B3(*state2list(x))
      preB4 = B4(*state2list(x))

      # success or fail
      if terminal:
        if i != args['test_episodes_len']-1:
          if np.sum(np.power(env.xk, 2)) < env.terminal_err:
            success_time += 1
          else:
            fail_time += 1
            fail_list.append((init_x, x))
          break
      elif i == args['test_episodes_len']-1:
        success_time += 1

    # epsoides results
    print 'state:\n', x, '\nlast action:', env.last_u
    print "----step: {} ----".format(i)
    print "-- lookaheadtimes: {} --".format(lookaheadtimes)

  # summary results
  print 'Success: {}, Fail: {}'.format(success_time, fail_time)
  print '#############Fail List:###############'
  for (i, e) in fail_list:
    print 'initial state: \n{}\nend state: \n{}\n----'.format(i, e)
  print 'shield times: {}, shield ratio: {}'.format(shield_count, float(shield_count)/(args['test_episodes_len']*args['test_episodes']))


  actor.sess.close()

if __name__ == "__main__":
  learning_eposides = int(sys.argv[1])
  actor_structure = [int(i) for i in list(sys.argv[2].split(','))]
  critic_structure = [int(i) for i in list(sys.argv[3].split(','))]
  train_dir = sys.argv[4]

  cartpole(learning_eposides, actor_structure, critic_structure, train_dir)
