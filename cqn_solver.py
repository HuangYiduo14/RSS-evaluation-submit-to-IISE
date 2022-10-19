import numpy as np
from agv_simulator_no_deadlock import WarehouseMap, new_workstation_param, MAX_T, Tt, Tdrop, Tload
from multiprocessing import Pool, cpu_count
import pandas as pd

def estimate_TH_cqn(num_AGV, setting='A',num_iter=100, input_flow_one_OD=0.0001,noflow=False,mg1=False,csv_name='temp.csv'):
    if noflow:
        num_iter =1
        input_flow_one_OD = 0

    for _ in range(num_iter):
        last_flow_one_OD = input_flow_one_OD
        #print(input_flow_one_OD)
        m, n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row = new_workstation_param(
            setting)
        warehouse = WarehouseMap(m, n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks,
                                 noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row)
        #speed_record, active_agv_number_record, fulfill_record, total_delay, total_delay_theory, total_delay_theory2, total_delay_det, block_flow, block_occ = warehouse.run_simulation(
        #    num_AGV=num_AGV, maxT=5000, use_sp=True)
        block_flow_theory, total_delay_theory, total_delay_det, total_delay_simp, deadlock_flag, block_rate = \
            warehouse.calc_sp_assignment(od_flow=input_flow_one_OD,maxT=MAX_T, num_AGV=num_AGV, get_block_rate=False)

        workstation_nodes = {ws: key for key, ws in enumerate(warehouse.ws_nodes)}
        dropoff_nodes = {warehouse.find_feasible_position_initial(*drop_block): key+len(workstation_nodes) for key, drop_block in enumerate(warehouse.dropoff_blocks)}

        Nw = len(workstation_nodes)
        Nd = len(dropoff_nodes)
        NM = Nw+Nd+Nw*Nd+Nd*Nw
        # construct p matrix
        P = np.zeros((NM,NM))
        # p_wd two process: w to trans, trans to d
        for i in range(Nw):
            for j in range(Nd):
                P[i,Nw+Nd+(i*Nd+j)] = 1./Nd
                P[Nw+Nd+(i*Nd+j), Nw+j] = 1.
        # p_dw
        for i in range(Nw):
            for j in range(Nd):
                P[Nw+j, Nw+Nd+Nw*Nd+(i*Nd+j)] = 1./Nw
                P[Nw+Nd+Nw*Nd+(i*Nd+j), i] = 1.
        # construct PA
        PA = P.copy()
        PA[:,0] = 0
        PA = np.eye(NM) - PA
        b_rhs = np.zeros(NM)
        b_rhs[0] = 1
        visit_ratio = np.linalg.solve(PA.T, b_rhs)


        warehouse_travel_info ={(a[0],a[1]): warehouse.od_travel_time_theory[a] for a in warehouse.od_travel_time_theory.keys()}
        # here, we have: warehouse.od_travel_time_theory[a] = [sp_length, total_delay_simp_route, total_delay_mg1_route]
        # a[0],a[1] is the origin, dest node
        warehouse_travel_info_int_index = {}
        for key, item in warehouse_travel_info.items():
            if (key[0] in workstation_nodes.keys()) and (key[1] in dropoff_nodes.keys()):
                warehouse_travel_info_int_index[(workstation_nodes[key[0]],dropoff_nodes[key[1]])] = item
            if (key[1] in workstation_nodes.keys()) and (key[0] in dropoff_nodes.keys()):
                warehouse_travel_info_int_index[(dropoff_nodes[key[0]], workstation_nodes[key[1]])] = item
                # each element is [0]: free-flow time, [1]: delay simple expected, [2]: delay mg1 expected
        demand_vect = visit_ratio.copy()
        for i in range(Nw):
            for j in range(Nd):
                if mg1:
                    demand_vect[Nw + Nd + (i * Nd + j)] = visit_ratio[Nw + Nd + (i * Nd + j)] * (
                                warehouse_travel_info_int_index[(i, Nw + j)][0] +
                                warehouse_travel_info_int_index[(i, Nw + j)][2])
                    demand_vect[Nw + Nd + Nw * Nd + (i * Nd + j)] = visit_ratio[Nw + Nd + Nw * Nd + (i * Nd + j)] * (
                                warehouse_travel_info_int_index[(Nw + j, i)][0] +
                                warehouse_travel_info_int_index[(Nw + j, i)][2])
                else:
                    demand_vect[Nw+Nd+(i*Nd+j)] = visit_ratio[Nw+Nd+(i*Nd+j)]*(warehouse_travel_info_int_index[(i,Nw+j)][0] + warehouse_travel_info_int_index[(i,Nw+j)][1])
                    demand_vect[Nw+Nd+Nw*Nd+(i*Nd+j)] = visit_ratio[Nw+Nd+Nw*Nd+(i*Nd+j)]*(warehouse_travel_info_int_index[(Nw+j,i)][0] +warehouse_travel_info_int_index[(Nw+j,i)][1])
        # MVA algorithm
        ENm = np.zeros(NM)
        ET_nu_m = np.zeros(NM)
        TH = 0

        for i in range(Nw):
            for j in range(Nd):
                ET_nu_m[Nd+Nw+(i*Nd+j)] = demand_vect[Nd+Nw+(i*Nd+j)]
                ET_nu_m[Nd+Nw+Nd*Nw+(i*Nd+j)] = demand_vect[Nd+Nw+Nd*Nw+(i*Nd+j)]
        for j in range(Nd):
            ET_nu_m[Nw+j] = Tdrop*visit_ratio[Nw+j]


        for n in range(1,num_AGV+1):
            for i in range(Nw):
                nu_i = visit_ratio[i]
                ET_nu_m[i] = (((Tload) ** 2) / 2 * nu_i * TH + (Tload) * (
                        ENm[i] + 1 - nu_i * TH * Tload
                )) * nu_i
            TH = n/np.sum(ET_nu_m)
            ENm = TH*ET_nu_m
        # calculate the actual input flow
        flow_list = visit_ratio*TH
        input_flow_one_OD = flow_list[0]/Nd
        if abs(input_flow_one_OD-last_flow_one_OD)<1e-7: #stop iteration if the difference is small enough
            _, _, _, _, _, block_rate = \
                warehouse.calc_sp_assignment(od_flow=input_flow_one_OD, maxT=MAX_T, num_AGV=num_AGV,
                                             get_block_rate=True)
            #import pdb;pdb.set_trace()
            break
        #print(input_flow_one_OD, deadlock_flag)

    print(num_AGV, 'AMVA TH:', input_flow_one_OD, 'noflow', noflow, 'is mg1?', mg1)

    df_block_theory = pd.DataFrame.from_dict(warehouse.block_flow_table_theory, orient='index',
                                             columns=['flow_turn_the', 'total_flow_the', 'delay_simp', 'delay_q',
                                                      'flow_drop'])
    df_block_theory['rho'] = 2 * df_block_theory['total_flow_the']/MAX_T + Tt * df_block_theory['flow_turn_the']/MAX_T +Tdrop* df_block_theory['flow_drop']/MAX_T
    df_block_theory.index = pd.MultiIndex.from_tuples(df_block_theory.index)
    #df_block_theory.to_csv(csv_name)
    #print('rho max',df_block_theory['rho'].max())
    return input_flow_one_OD, total_delay_theory, total_delay_simp, df_block_theory['rho'].max(), df_block_theory['rho'].mean(), np.max(block_rate)


# result 2:
def do_simulation(num_AGV, setting='A', csv_name='temp.csv'):
    m, n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row = new_workstation_param(
        setting)
    warehouse = WarehouseMap(m, n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks,
                             allow_up_col, allow_down_col, allow_left_row, allow_right_row)
    workstation = {ws: key for key, ws in enumerate(warehouse.workstation_exit_blocks)}
    Nw = len(workstation)
    dropoff_points = {ws: key + Nw for key, ws in enumerate(warehouse.dropoff_blocks)}
    Nd = len(dropoff_points)
    speed_record, active_agv_number_record, fulfill_record, total_delay,total_delay_theory,total_delay_theory2,total_delay_det, block_flow, block_occ, deadlock_flag,\
        one_one_blocking_chance, complicated_blocking_chance, complicated_blocking_rate = \
        warehouse.run_simulation(num_AGV=num_AGV, maxT=MAX_T, use_sp=True)
    complicated_blocking_rate = complicated_blocking_rate*3600

    flow_fulfill = max(fulfill_record)
    print('simulated fulfillment rate',flow_fulfill/Nw/Nd/MAX_T)
    # store simulation results
    df_block = pd.DataFrame.from_dict(block_flow, orient='index', columns=['flow_turn', 'flow_straight', 'delay'])
    block_occ_dict = {(i, j): [block_occ[i, j]] for i in range(10) for j in range(10)}
    df_block_occ = pd.DataFrame.from_dict(block_occ_dict, orient='index', columns=['occ'])
    df_block_join = df_block.join(df_block_occ)
    df_block_theory = pd.DataFrame.from_dict(warehouse.block_flow_table_theory, orient='index',
                                             columns=['flow_turn_the', 'total_flow_the', 'delay_simp', 'delay_q',
                                                      'delay_simp_2'])
    df_block_join = df_block_join.join(df_block_theory)
    df_block_join.index = pd.MultiIndex.from_tuples(df_block_join.index)
    df_block_join['rho'] = 2 * df_block_join['total_flow_the'] / MAX_T + Tt * df_block_join['flow_turn_the'] / MAX_T
    #df_block_join.to_csv(csv_name)
    max_rho = df_block_join['rho'].max()
    avg_rho = df_block_join['rho'].mean()
    return flow_fulfill, Nw, Nd, np.mean(speed_record), total_delay, deadlock_flag, one_one_blocking_chance, complicated_blocking_chance, complicated_blocking_rate

def return_TH(num_AGV_setting):
    num_AGV = num_AGV_setting[0]
    setting = num_AGV_setting[1]
    TH1_list = []
    agv_speed_list = []
    total_delay_list = []
    deadlock_sim_list = []
    complicated_blocking_rate_list =[]
    for i in range(exp_num_per_agv):
        TH1, Nw, Nd, avg_speed, total_delay, deadlock_flag, one_one_blocking_chance, complicated_blocking_chance, complicated_blocking_rate= \
            do_simulation(num_AGV, setting=setting, csv_name=setting+str(num_AGV)+'simulation_result_{0}.csv'.format(i))
        TH1_list.append(TH1)
        agv_speed_list.append(avg_speed)
        total_delay_list.append(total_delay)
        deadlock_sim_list.append(deadlock_flag)
        complicated_blocking_rate_list.append(complicated_blocking_rate)
    TH2, total_delay_theory2, total_delay_simp2, maxrho_2, avg_rho2, block_rate_2 = estimate_TH_cqn(setting=setting, num_AGV=num_AGV, csv_name=setting+str(num_AGV)+'renewal_cqn.csv') # renewal process delay
    TH2 = TH2 * MAX_T* Nw *Nd
    TH3, total_delay_theory3, total_delay_simp3,maxrho_3, avg_rho3, block_rate_3= estimate_TH_cqn(setting=setting, num_AGV=num_AGV,mg1=True,csv_name=setting+str(num_AGV)+'mg1_cqn.csv') # mg1 delay
    TH3 = TH3 * MAX_T*Nw*Nd
    TH4, total_delay_theory4, total_delay_simp4,maxrho_4, avg_rho4, block_rate_4 = estimate_TH_cqn(setting=setting, num_AGV=num_AGV,noflow=True, csv_name=setting+str(num_AGV)+'no_delay.csv') # no delay
    TH4 = TH4 * MAX_T*Nw*Nd
    return TH1_list+[TH2,TH3,TH4]+total_delay_list+[total_delay_simp2,total_delay_theory3,total_delay_simp4]+agv_speed_list+deadlock_sim_list+[maxrho_2,maxrho_3,maxrho_4]+[avg_rho2,avg_rho3,avg_rho4]+\
           [np.max(block_rate_2),np.max(block_rate_3),np.max(block_rate_4)]+complicated_blocking_rate_list

if __name__ == '__main__':
    max_agv_num = 22
    exp_num = 22
    agv_list = [max(1,int(max_agv_num*i/exp_num)) for i in range(1,exp_num+1)]
    setting_list = ['A','B','C']
    # do simulations
    exp_num_per_agv = 100
    for setting in setting_list:
        print(setting)
        agv_list_setting = [(agv,setting) for agv in agv_list]
        results = []
        #results.append(return_TH(agv_list_setting[1]))
        #for agv_setting in agv_list_setting:
        #    results.append(return_TH(agv_setting))
        with Pool(5) as p:
           results = p.map(return_TH, agv_list_setting)
        #results.append(return_TH((2, 'C')))
        results = pd.DataFrame(results,columns=['simulation TH '+str(i) for i in range(exp_num_per_agv)]
                                               +['renewal process TH','M/G/1 TH','no congestion TH']
                                               +['simulation delay '+str(i) for i in range(exp_num_per_agv)]
                                               +['delay renewal','delay mg1', 'delay no cong']
                                               +['simulation speed '+str(i) for i in range(exp_num_per_agv)]
                               +['deadlock sim'+str(i) for i in range(exp_num_per_agv)]
                                               +['max rho renewal','max rho mg1', 'max rho no cong']
                                               +['avg rho renewal','avg rho mg1', 'avg rho no cong']
                                               +['block rate renewal','block rate mg1','block rate no cong']
                                               +['block rate sim'+str(i) for i in range(exp_num_per_agv)])
        results.to_csv(setting+'_compare_TH2.csv')
    # do evaluations
    exp_num_per_agv = 1
    for setting in setting_list:
        print(setting)
        agv_list_setting = [(agv,setting) for agv in agv_list]
        results = []
        #results.append(return_TH(agv_list_setting[1]))
        #for agv_setting in agv_list_setting:
        #    results.append(return_TH(agv_setting))
        with Pool(5) as p:
           results = p.map(return_TH, agv_list_setting)
        #results.append(return_TH((2, 'C')))
        results = pd.DataFrame(results,columns=['simulation TH '+str(i) for i in range(exp_num_per_agv)]
                                               +['renewal process TH','M/G/1 TH','no congestion TH']
                                               +['simulation delay '+str(i) for i in range(exp_num_per_agv)]
                                               +['delay renewal','delay mg1', 'delay no cong']
                                               +['simulation speed '+str(i) for i in range(exp_num_per_agv)]
                               +['deadlock sim'+str(i) for i in range(exp_num_per_agv)]
                                               +['max rho renewal','max rho mg1', 'max rho no cong']
                                               +['avg rho renewal','avg rho mg1', 'avg rho no cong']
                                               +['block rate renewal','block rate mg1','block rate no cong']
                                               +['block rate sim'+str(i) for i in range(exp_num_per_agv)])
        results.to_csv(setting+'_compare_TH2.csv')
        
