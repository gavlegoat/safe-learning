Artifact Evaluation for Paper #291
==================================

An Inductive Synthesis Framework for Verifiable Machine Learning
==================================


We provide a prepared docker file to run our artifact. The experiment result may be different than what we report in the paper because (1) it is a docker environment with limited memory (2) affected by the environment, the tool in the docker file may have different behaviors, i.e., the random generator used by the TensorFlow library is different, which may lead to a significantly different search direction of both programs and invariants in our search space. However, the artifact suffices to prove the reproducibility of our work.

Please follow the steps below to get our artifact:

# Getting Started Guide: 
1. Install Docker:  
[https://docs.docker.com/get-started/](https://docs.docker.com/get-started/)

2. Pull Verifiable Reinforcement Learning Environment docker image:   
```  
docker pull caffett/vrl_env
```
3. Start docker environment:  
```  
docker run -it caffett/vrl_env:v0 /bin/bash
```
4. Clone VRL code:  
```
git clone git@github.com:caffett/VRL_CodeReview.git
```

# Step-by-Step Instructions
## Run Command
Pretrained neural networks are provided in our artifact. We do not provide an interface to retrain neural network because retraining requires significant manual efforts to adjust a number of training parameters and is time-consuming. The path to these pretrained neural models is `ddpg_chkp/<model_name>/<network_structure>`.

Our tool provides a python script for each of our benchmarks. Given a benchmark, the user just need to type:

```
python <benchmark_name> 
[--nn_test | --shield_test | --retrain_shield | --test_episodes=TEST_EPISODES]
``` 

There are 4 flags: 
**--nn\_test**: adding this flag runs the pretrained neural network controller alone without shield protection.  
**--shield\_test**:  adding this flag runs the pretrained neural network controller in tandem with a pre-synthesized and verified program to provide shield protection.  
**--retrain\_shield**: adding this flag re-synthesizes a deterministic program that is also verified safe.
**--test\_episodes**: This parameter specifies the number of steps used for each simulation run. The default value is 100.    

## Getting Results
### Run a Single Branch-mark
After running a simulation of a benchmark, our tool reports the total simulation time and the number of times that a system enters an unsafe region.  
For example, running `python 4-car-platoon.py --nn_test --test_episodes=1000` might produce the following result:

<center>
![](https://user-images.githubusercontent.com/11462215/53280122-bcd48b00-36e4-11e9-83aa-fa171fe74e7c.png)
![](https://user-images.githubusercontent.com/11462215/53280155-18067d80-36e5-11e9-9a9f-3a767f0b12f3.png)
</center>

The system is unsafe since a number of safety violations are observed.

Running a neural network controller in tandem with a verified program distilled from it can eliminate those unsafe neural actions. Our tool produces in the output the number of interventions from the program to the neural network controller. It also gives the total running time.

Running `python 4-car-platoon.py --shield_test --test_episodes=1000` might produce the following result (using a pre-synthesized and verified program):
<center>
![image](https://user-images.githubusercontent.com/11462215/53280233-d1fde980-36e5-11e9-8e7a-82111927ad56.png)
</center>

Based on the neural network simulation time and shield simulation time, we can calculate the overhead of using a shield.  

```
Overhead = (shield_test_runing_time - neural_network_test_runing_time) /
neural_network_test_runing_time * 100%
```
  
For each benchmark, with the protection of a shield, our system never enters an unsafe region. We may get the following result for all benchmarks.

<center>
![image](https://user-images.githubusercontent.com/11462215/53280265-21dcb080-36e6-11e9-9ed7-1a9146b6529e.png)
</center>

Running with --retrain_shield can re-synthesize a new deterministic program to replace the original one. After re-synthesis, our tool produces the total synthesis time.  For example, we may get the following result by running `python 4-car-platoon.py --retrain_shield --test_episodes=1000`. 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280299-6f591d80-36e6-11e9-88d3-0b83c97dec26.png)
</center>

We count how many iterations our verification algorithm needs to synthesize a deterministic program and this result corresponds to the size of a re-synthesized program (i.e., the number of branches in a synthesized program). 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280317-a5969d00-36e6-11e9-86d7-0a13f31b1c57.png)
</center>

### Run All Branch-marks  

We also provide some scripts to run all of our benchmarks in a batch mode:

`./run_test_100ep.sh`: Run all the benchmarks with pretrained neural networks. The number of steps used in each simulation run is set to 100.     
`./run_test_1000ep.sh`: Run all the benchmarks with pretrained neural networks. The number of steps used in test simulation run is set 1000.    
`./run_retrain_shield.sh`: Retrain deterministic programs for all the benchmarks.
