import pandas as pd
import numpy as np
exp_record = []
fp = open('out.txt','r')
for line in fp.readlines():
    if len(line)>27 and line[:27]=='chance of one-one blocking ':
        p_one_one = float(line[27:])
        data_line = previous_line.split(' ')
        agv_number = float(data_line[0])
        sim_delay = float(data_line[3])
        estimate_delay = float(data_line[4])
        if sim_delay>estimate_delay*2 or sim_delay/agv_number>3000:
            is_sat = True
        else:
            is_sat = False
    if len(line)>32 and line[:32]=='chance of non-independent queue ':
        p_complicated = float(line[32:])
    if len(line)>30 and line[:30]=='rate of non-independent queue ':
        rate_complicated = float(line[30:])
        p_one_bottleneck = 1- p_one_one-p_complicated
        exp_record.append([agv_number, is_sat, sim_delay,p_one_one, p_complicated, p_one_bottleneck, rate_complicated])

    previous_line = line
fp.close()

record = pd.DataFrame(exp_record, columns=['R','is_sat','total delay','one to one prob','comp prob','on bottleneck prob','comp rate'])
record_nonsat = record.loc[record['is_sat']==False]
record_by_R =  record_nonsat.groupby('R').agg('mean')
record_by_R.to_csv('prob sat.csv')