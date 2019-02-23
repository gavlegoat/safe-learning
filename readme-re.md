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
## Run Model Command
There are some models have already been trained. The path stores them are `ddpg_chkp/<model_name>/<network_structure>`.

```
python <branchmark_name> 
[--nn_test | --shield_test | --retrain_shield | --test_episodes=TEST_EPISODES]
``` 
will run the model trained model.
There are 4 flags for these branch-marks  
**--nn\_test**: adding this flag will run the test with trained neural network controller but without shield protection.  
**--shield\_test**:  adding this flag will run the test with shield protection.  
**--retrain\_shield**: adding this flag will retrain the shield.  
**--test\_episodes**: This parameter is used for assigning the number of episodes for test. The default value of this parameter is 100.    

## Getting Results
### Run a Single Branch-mark
After running the neural network test, we can get the time for running this test and the times that our system enters the unsafe region.  
For example, when we run `python 4-car-platoon.py --nn_test --test_episodes=1000`, we will get the following results.  

<center>
![](https://user-images.githubusercontent.com/11462215/53280122-bcd48b00-36e4-11e9-83aa-fa171fe74e7c.png)
![](https://user-images.githubusercontent.com/11462215/53280155-18067d80-36e5-11e9-9a9f-3a767f0b12f3.png)
</center>


Similarly, running the shield test can get the number of interventions, and the time for running shield test.  
We can get the following result by running `python 4-car-platoon.py --shield_test --test_episodes=1000`.  
<center>
![image](https://user-images.githubusercontent.com/11462215/53280233-d1fde980-36e5-11e9-8e7a-82111927ad56.png)
</center>

According to the neural network test time and shield test time, we can calculate the overhead of using shield.  

```
Overhead = (shield_test_runing_time - neural_network_test_runing_time) /
neural_network_test_runing_time * 100%
```
  
For each branch-mark, since with the protection of shield, our system will never enter an unsafe region. Therefore, we can get following results for all branch-marks.

<center>
![image](https://user-images.githubusercontent.com/11462215/53280265-21dcb080-36e6-11e9-9ed7-1a9146b6529e.png)
</center>

When running with --retrain_shield, the running model will re-synthesize the shield from scratch. After re-synthesizing the shield, we can get the synthesis time.  
We can get the following result by running `python 4-car-platoon.py --retrain_shield --test_episodes=1000`. 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280299-6f591d80-36e6-11e9-88d3-0b83c97dec26.png)
</center>

Count how many times verification algorithm finds the controller, the result is the size of the re-synthesized shield. 
<center>
![image](https://user-images.githubusercontent.com/11462215/53280317-a5969d00-36e6-11e9-86d7-0a13f31b1c57.png)
</center>

### Run All Branch-marks  
`./run_test_100ep.sh`: Run all branch-marks test with trained model, the test episode is 100.     
`./run_test_1000ep.sh`: Run all branch-marks test with trained model, the test episode is 1000.    
`./run_retrain_shield.sh`: retrain shield for all branch-marks.       
