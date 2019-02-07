# @Author: Zikang Xiong
# @Date:   2018-10-22 20:41:12
# @Last Modified by:   Zikang Xiong
# @Last Modified time: 2018-10-22 22:29:22
for i in $(seq 50 50 1000);
do
	mkdir ddpg_chkp/pendulum/continuous/$(($i * 6/5))$i$(($i * 7/5))$(($i * 6/5))$i; 

	python pendulum_continuous_ddpg.py\
	 $((300+$i))\
	 $(($i * 6/5)),$i\
	 $(($i * 7/5)),$(($i * 6/5)),$i\
	 ddpg_chkp/pendulum/continuous/$(($i * 6/5))$i$(($i * 7/5))$(($i * 6/5))$i/\
	 >> ddpg_chkp/pendulum/continuous/$(($i * 6/5))$i$(($i * 7/5))$(($i * 6/5))$i/log;
done