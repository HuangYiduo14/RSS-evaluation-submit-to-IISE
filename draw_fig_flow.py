import pandas as pd
import matplotlib.pyplot as plt
from agv_simulator_no_deadlock import MAX_T

MAX_T1 = 100
MAX_T2 = 10000
from cqn_solver import exp_num_per_agv
EPS = 1e-12
exp_setting = 'C'
exp_num_per_agv=100

data0= pd.read_csv(exp_setting+'_compare_TH.csv')
data0.rename({'Unnamed: 0': 'R'}, axis=1, inplace=True)
data0['R'] = data0['R'] + 1

data1 = pd.read_csv(exp_setting+'_compare_TH2.csv')
data1.rename({'Unnamed: 0': 'R'}, axis=1, inplace=True)
data1['R'] = data0['R'] + 1

update_cols = ['delay renewal','delay mg1', 'delay no cong']+['renewal process TH','M/G/1 TH','no congestion TH']+\
              ['max rho renewal','max rho mg1', 'max rho no cong']+['avg rho renewal','avg rho mg1', 'avg rho no cong']+\
              ['block rate renewal','block rate mg1','block rate no cong']
for col in update_cols:
    data0[col] = data1[col]

deadlock_prob = []
plt.figure(1) #agv-flow plot
data_th = []
for index in data0.index:
    max_TH = max([data0.loc[index, 'simulation TH ' + str(i)]/MAX_T2  for i in range(exp_num_per_agv)])
    data_th_i = [data0.loc[index, 'simulation TH ' + str(i)]/MAX_T2  for i in range(exp_num_per_agv) if data0.loc[index, 'simulation TH ' + str(i)]/MAX_T>0.*max_TH]
    deadlock_prob.append(1. - len(data_th_i)/exp_num_per_agv)
    data_th.append(data_th_i)

plt.boxplot(data_th, positions=data0['R'].values,showfliers=False)
for i in range(exp_num_per_agv):
    if i==0:
        plt.scatter(data0['R'], data0['simulation TH ' + str(i)] / MAX_T2, color='blue',label='SIM', alpha=0.5, s=1)
    else:
        plt.scatter(data0['R'], data0['simulation TH ' + str(i)] / MAX_T2, color='blue', alpha=0.5,s=1)


#data0.loc[data0['max rho renewal']>1-EPS,'renewal process TH'] =  0.
#data0.loc[data0['max rho mg1']>1-EPS,'M/G/1 TH'] =  0.

plt.plot(data0['R'],data0['renewal process TH']/MAX_T1,color='red',label='RP-CQN')
plt.plot(data0['R'],data0['M/G/1 TH']/MAX_T1,color='black',label='MG1-CQN')
plt.plot(data0['R'],data0['no congestion TH']/MAX_T1 ,color='green',label='CF-CQN')
plt.xlabel('number of AGV')
plt.ylabel('throughput (AGV per unit time)')
plt.legend()
plt.savefig(exp_setting+'_th-density.png',dpi=400)

plt.figure(2) #agv-delay plot
data_th = []
for index in data0.index:
    max_TH = max([data0.loc[index, 'simulation TH ' + str(i)]/MAX_T2  for i in range(exp_num_per_agv)])
    data_th_i = [data0.loc[index, 'simulation delay '+str(i)]/MAX_T2  for i in range(exp_num_per_agv) if data0.loc[index, 'simulation TH ' + str(i)]/MAX_T>0.*max_TH]
    data_th.append(data_th_i)
plt.boxplot(data_th, positions=data0['R'].values,showfliers=False)
for i in range(exp_num_per_agv):
    if i==0:
        plt.scatter(data0['R'], data0['simulation delay '+str(i)] / MAX_T2, color='blue',label='SIM', alpha=0.5, s=1)
    else:
        plt.scatter(data0['R'], data0['simulation delay '+str(i)] / MAX_T2, color='blue', alpha=0.5, s=1)
#data0.loc[data0['max rho renewal']>1-EPS,'delay renewal'] =  40 *MAX_T
#data0.loc[data0['max rho mg1']>1-EPS,'delay mg1'] = 40*MAX_T

plt.plot(data0['R'],data0['delay renewal']/MAX_T1,color='red',label='RP-CQN')
plt.plot(data0['R'],data0['delay mg1']/MAX_T1,color='black',label='MG1-CQN')
plt.plot(data0['R'],data0['delay no cong']/MAX_T1 ,color='green',label='CF-CQN')
plt.ylim(0,22)
plt.xlabel('number of AGV')
plt.ylabel('delay rate (unit time per unit time)')
plt.legend()
plt.savefig(exp_setting+'delay-density.png',dpi=400)


fig=plt.figure(3) #flow-speed
fig.set_size_inches(3, 2)
for i in range(exp_num_per_agv):
    if i==0:
        plt.scatter(data0['simulation TH ' + str(i)] / MAX_T2, data0['simulation speed '+str(i)], color='blue',label='SIM',s=2)
    else:
        plt.scatter(data0['simulation TH ' + str(i)] / MAX_T2, data0['simulation speed '+str(i)], color='blue',s=2)
plt.xlabel('throughput (AGV per unit time)')
plt.ylabel('average speed (unit distance per unit time)')
plt.savefig(exp_setting+'speed-th.png',dpi=400)


fig = plt.figure(4) #density-speed
fig.set_size_inches(3, 2)
for i in range(exp_num_per_agv):
    if i==0:
        plt.scatter(data0['R'] , data0['simulation speed '+str(i)], color='blue',label='SIM',s=2)
    else:
        plt.scatter(data0['R'] , data0['simulation speed '+str(i)], color='blue',s=2)
plt.xlabel('number of AGV')
plt.ylabel('average speed (unit distance per unit time)')
plt.legend()

plt.savefig(exp_setting+'speed-density.png',dpi=400)

plt.figure(5)


data0['max rho renewal 1'] = data0['max rho renewal']
data0.loc[data0['max rho renewal']>=1,'max rho renewal 1'] = 1
data0['max rho mg1 1'] = data0['max rho mg1']
data0.loc[data0['max rho mg1']>=1,'max rho mg1 1'] = 1

#plt.plot(data0['R'],data0['max rho renewal 1'],color='red',label='RP-CQN max rho')
#plt.plot(data0['R'],data0['max rho mg1 1'],color='black',label='MG1-CQN max rho')




data_block_rate = []
for index in data0.index:
    data_block_rate_i = [data0.loc[index, 'block rate sim' + str(i)]  for i in range(exp_num_per_agv)]
    data_block_rate.append(data_block_rate_i)

plt.boxplot(data_block_rate, positions=data0['R'].values,showfliers=False)
plt.plot(data0['R'],data0['block rate renewal']*3600,color='red',label='RP-CQN',linestyle='solid')
plt.plot(data0['R'],data0['block rate mg1']*3600,color='black',label='MG1-CQN',linestyle='solid')
plt.xlim(1,14)
plt.ylim(0,40)
for i in range(exp_num_per_agv):
    if i==0:
        plt.scatter(data0['R'], data0['block rate sim'+str(i)], color='blue',label='SIM', alpha=0.5, s=1)
    else:
        plt.scatter(data0['R'], data0['block rate sim'+str(i)], color='blue', alpha=0.5, s=1)

#plt.plot(data0['R'], deadlock_prob, label='Prob(deadlock) of SIM',color='blue')

#plt.scatter([14],data0['max rho mg1'][13],color='black', label='(14,{0:.4f})'.format(data0['max rho mg1 1'][13]))
#plt.scatter([14],deadlock_prob[13],color='blue', label='(14,{0})'.format(deadlock_prob[13]))
plt.xlabel('number of AGV')
plt.ylabel('complicated blocking rate per hour')
plt.legend(loc='best', bbox_to_anchor=(0.3,1))
plt.savefig(exp_setting+'util-prob.png',dpi=400)


