env_name=cartpole_continous
script_name=cartpole_continuous_shield.py
training_start_step=100
steps=10
training_ended_step=50000
actor_structure=300,200
critic_structure=300,250,200

mkdir ddpg_chkp/
mkdir ddpg_chkp/auto/
mkdir ddpg_chkp/auto/$env_name/

for iter in $(seq $training_start_step $steps $training_ended_step)
do
	mkdir ddpg_chkp/auto/$env_name/$iter
	python $script_name $iter $actor_structure $critic_structure ddpg_chkp/auto/$env_name/$iter/ > ddpg_chkp/auto/$env_name/$iter/log 2>&1
done