import networkx as nx
import numpy as np
import random
import copy
import warnings
import time
#import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt

import math
import torch
from torch.autograd import grad
from scipy.stats import binom

Tt=4
Tdrop = 1
Tload = 5
S1 = 2
S2 = 2+Tt
S3 = 2+Tdrop

BIG_NUM = 999999
MAX_CAP = 10000
MAX_T = 10000
# 5/19: no collision at OD
# no collision around
def m_distance(node1,node2):
    return abs(node2-node1.x)+abs(node2.y-node1.y)

def nth_derivative(fun, wrt, n):
    for i in range(n):
        if not fun.requires_grad:
            return torch.zeros_like(wrt)
        grads = grad(fun, wrt, create_graph=True)[0]
        fun = grads.sum()
    return grads

class AGV:
    def __init__(self, current_node, dest_node, unique_index, t):
        self.current_node = current_node
        self.origin_node = current_node
        self.previous_node = current_node
        self.dest_node = dest_node
        self.countdown = 0
        self.next_node = current_node
        self.path = []
        self.load = 0 #load 0: going to workstation, 1: going to dropoff point, 2: loading, 3: unloading, 4: idle
        self.unique_index = unique_index
        self.delay_time = 0
        self.created_time = t
        self.current_task_start_time = t
        self.ws_nodes = [] # set of nodes serving as workstations

    def print_s(self):
        print(self.unique_index,'node:', self.current_node, 'dest:', self.dest_node, 'countdown:',self.countdown)
        print(self.path)
        print('next node:',self.next_node, 'load:',self.load)

    def check_task_complete(self):
        return self.current_node == self.dest_node

    def assigne_new_dest(self, new_dest, t, change_origin_info = False):
        if change_origin_info:
            self.origin_node = self.current_node
            self.current_task_start_time = t
        self.dest_node = new_dest
        self.countdown = 0
        self.path = []

    def load_item(self, load_time=1):
        self.load = 2
        self.countdown = load_time
        self.next_node = self.current_node

    def unload_item(self,unload_time = 1):
        self.load = 3
        self.countdown = unload_time
        self.next_node = self.current_node

    def move_to_dest(self):
        self.current_node = self.dest_node
        self.countdown = 1
        self.previous_node = self.dest_node
        self.next_node = self.dest_node
        self.path = [self.dest_node]
        self.origin_node = self.dest_node

class WarehouseMap:  # m n must be even to have a connected graph
    def __init__(self, m, n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row):
        self.caltime0=time.time()
        self.M = m
        self.N = n  # M*N network, grid index, m is the width, n is the height, i is the height index, j is the width index
        # in the network, one square = 4 nodes, index of node = 4*(M*i+j)+k, (i<N,j<M, k=0,1,2,3), 0:up, 1:right, 2:down, 3:left
        self.map_allow_dir = np.zeros((m, n, 4))  # [up, right, down, left]
        # assign direction to blocks
        for j in allow_up_col:
            for i in range(n):
                self.map_allow_dir[j][i][0] = 1
        for j in allow_down_col:
            for i in range(n):
                self.map_allow_dir[j][i][2] = 1
        for j in range(m):
            for i in allow_right_row:
                self.map_allow_dir[j][i][1] = 1
        for j in range(m):
            for i in allow_left_row:
                self.map_allow_dir[j][i][3] = 1
        # noaccess blocks shouldn't be allowed
        for block in noaccess_blocks:
            self.map_allow_dir[block[1]][block[0]] = 0
        # initialize all nodes on the graph
        self.graph = nx.DiGraph()
        pos = dict()
        div_k = {0: [0, 0.4], 1: [0.4, 0], 2: [0, -0.4], 3: [-0.4, 0]}  # only for plot [up, right, down, left]
        self.node_div_k = div_k
        for i in range(n):
            for j in range(m):
                for k in range(4):
                    self.graph.add_node(4 * (m * i + j) + k, x=j, y=i)
                    pos[4 * (m * i + j) + k] = (j + div_k[k][0], i + div_k[k][1])
        # add workstation nodes
        for i in range(len(workstation_enter_blocks)):
            self.graph.add_node(m*n*4+i)
            if workstation_enter_blocks[i][1]==0:
                pos[m*n*4+i] = (-1, workstation_enter_blocks[i][0]+0.5)
            else:
                pos[m * n * 4 + i] = (m, workstation_enter_blocks[i][0]+0.5)

        # connect 0:up, 1:right, 2:down, 3:left nodes
        for j in allow_up_col:
            for i in range(n - 1):
                self.graph.add_edge(4 * (m * i + j), 4 * (m * (i + 1) + j), dist=1, weight=1,
                                    flow=0, delay=0, type='between')
        for j in allow_down_col:
            for i in range(n-1):
                self.graph.add_edge(4 * (m * (i+1) + j)+2, 4 * (m *i + j)+2, dist=1, weight=1,
                                    flow=0, delay=0, type='between')
        for j in range(m - 1):
            for i in allow_left_row:
                self.graph.add_edge(4 * (m * i + j+1)+3, 4 * (m * i + j)+3, dist=1, weight=1, flow=0,
                                    delay=0, type='between')
        for j in range(m - 1):
            for i in allow_right_row:
                self.graph.add_edge(4 * (m * i + j) + 1, 4 * (m * i + j+1) + 1, dist=1, weight=1, flow=0,
                                    delay=0, type='between')
        for i in range(n):
            for j in range(m):
                #import pdb; pdb.set_trace()
                allowed_dir = np.where(self.map_allow_dir[j][i] == 1)
                allowed_dir = allowed_dir[0]
                allowed_dir.sort()
                for i1 in allowed_dir:
                    for j1 in allowed_dir:
                        if i1 != j1:
                            self.graph.add_edge(4 * (m * i + j) + i1, 4 * (m * i + j) + j1, dist=Tt, weight=Tt, flow=0,
                                                delay=0, type='in')
        # connect workstation nodes
        for i in range(len(workstation_enter_blocks)):
            if workstation_enter_blocks[i][1] == 0:
                self.graph.add_edge(4 * (m * workstation_enter_blocks[i][0] + workstation_enter_blocks[i][1]) + 3,
                                    m * n * 4 + i, dist=1, weight=1, flow=0, delay=0, type='to ws')
                self.graph.add_edge(m * n * 4 + i,
                                    4 * (m * workstation_exit_blocks[i][0] + workstation_exit_blocks[i][1]) + 1,
                                    dist=1, weight=1, flow=0, delay=0, type='from ws')
            else:
                self.graph.add_edge(4 * (m * workstation_enter_blocks[i][0] + workstation_enter_blocks[i][1]) + 1,
                                    m * n * 4 + i, dist=1, weight=1, flow=0, delay=0, type='to ws')
                self.graph.add_edge(m * n * 4 + i,
                                    4 * (m * workstation_exit_blocks[i][0] + workstation_exit_blocks[i][1]) + 3,
                                    dist=1, weight=1, flow=0, delay=0, type='from ws')
        self.ws_nodes = [m*n*4+i for i in range(len(workstation_enter_blocks))]

        self.pos = pos
        #draw network
        #fig, axe = plt.subplots(1,1)
        #dict_nodes = dict(self.graph.degree)
        #nx.draw(self.graph,ax=axe, nodelist = list(dict_nodes.keys()), arrowsize = 8, pos=pos, with_labels=False,node_size=[(v>0) * 100 for v in dict_nodes.values()])
        #for i in range(n+1):
        #    plt.plot([-0.5,m-0.5],[i-0.5,i-0.5],color='blue')
        #for j in range(m+1):
        #    plt.plot([j-0.5,j-0.5],[-0.5,n-0.5],color='blue')
        #fig.set_size_inches(20, 20)
        #fig.savefig('test2png.png', dpi=300)
        #plt.show()
        self.node_pos = pos
        self.init_nodes = []
        self.dest_nodes = []
        self.od_travel_time = dict()
        self.od_travel_time_theory = dict()
        self.workstation_enter_blocks = workstation_enter_blocks
        self.workstation_exit_blocks = workstation_exit_blocks
        self.dropoff_blocks = dropoff_blocks
        self.noaccess_blocks = noaccess_blocks
        self.agv_fleet = []
        self.deadlock_graph = nx.DiGraph()
        self.pmf_queue = dict()
        return

    def find_feasible_position_initial(self, i, j):
        # location i,j to node index
        for k in range(4):
            if (self.graph.out_degree(4 * (self.M * i + j) + k) > 0) and (self.graph.in_degree(4 * (self.M * i + j) + k)>0):
                return 4 * (self.M * i + j) + k
        warnings.warn('no feasible point')
    def cell2nodes(self, i, j):
        return [4 * (self.M * i + j) + k for k in range(4) if (self.graph.out_degree(4 * (self.M * i + j) + k) > 0) and (self.graph.in_degree(4 * (self.M * i + j) + k)>0)]

    def cdf_first_3(self, i, j, f1, f2, f3):
        pmf_list = [0 for i in range(4)]
        if (i,j) in self.pmf_queue.keys():
            return self.pmf_queue[(i,j)]
        if f1+f2+f3<1e-10:
            self.pmf_queue[(i, j)] = pmf_list
            return pmf_list

        f = f1 + f2 + f3
        rho = S1 * f1 + S2 * f2 + S3 * f3
        p1 = f1 / f
        p2 = f2 / f
        p3 = f3 / f

        x = torch.tensor(0., requires_grad=True)
        pi = (1. - x) * (1. - rho) * (
                    p1 * torch.exp(-S1 * (f * (1. - x))) + p2 * torch.exp(-S2 * (f * (1. - x))) + p3 * torch.exp(
                -S3 * (f * (1. - x)))) / \
             ((p1 * torch.exp(-S1 * (f * (1. - x))) + p2 * torch.exp(-S2 * (f * (1. - x))) + p3 * torch.exp(
                 -S3 * (f * (1. - x)))) - x)

        for n in range(4):
            if n == 0:
                pmf_list[n] = float(pi)
            else:
                pmf_list[n] = float(nth_derivative(pi, x, n)) / math.factorial(n)
        for i in range(len(pmf_list)):
            if math.isnan(pmf_list[i]):
                pmf_list[i]=0
            if pmf_list[i]<0:
                pmf_list[i]=0

        self.pmf_queue[(i,j)] = pmf_list

        #print(pmf_list)
        return pmf_list
    def node2loc(self, node_index):
        # node index to location i,j
        if node_index < self.M*self.N*4:
            node_loc_index = node_index // 4
            return int(node_loc_index // self.M), int(node_loc_index % self.M)
        else:
            return -999,-999

    def node_m_distance(self, node1, node2):
        # manhattan distance, can be used in A*
        x1, y1 = int(node1 // self.M), int(node1 % self.M)
        x2, y2 = int(node2 // self.M), int(node2 % self.M)
        return abs(x1-x2)+abs(y1-y2)

    def generate_one_AGV(self):
        # randomly select one empty and accessible block
        if len(self.agv_fleet)>0:
            occupied_blocks = np.array([agv.current_node for agv in self.agv_fleet]) // 4
        else:
            occupied_blocks = []
        nonocc_blocks = set([self.M * i + j for i in range(self.N) for j in range(self.M)]) - set(
                occupied_blocks) - set([self.M*block[0]+block[1] for block in self.noaccess_blocks])
        if len(nonocc_blocks) == 0:
            print("no more empty blocks, deadlock")
            #import pdb; pdb.set_trace()
            return -999, -999
        else:
            init_block_index = random.sample(list(nonocc_blocks), 1)
            init_block_index = init_block_index[0]
            #print('new agv loc', int(init_block_index // self.M), int(init_block_index % self.M))
            return int(init_block_index // self.M), int(init_block_index % self.M)

    def run_simulation(self, maxT=1000, num_AGV = 10, use_sp=True):
        # initialize
        deadlock_flag = 0
        m = self.M
        n = self.N

        total_num_queues = 0
        total_num_complicated_queues = 0
        total_num_simple_blocking = 0

        self.dropoff_map = np.zeros((n,m))
        self.block_map = np.zeros((n,m))
        self.block_flow_table = {(i,j):[0,0,0] for i in range(n) for j in range(m)}
        fulfilled = 0
        fulfill_record = []
        speed_record = []
        active_agv_number_record = []
        total_delay = 0
        agv_number_max = 0
        # randomly generate AGVs
        for _ in range(num_AGV):
            new_agv_pos = self.generate_one_AGV()
            ori_node = self.find_feasible_position_initial(*new_agv_pos)
            new_agv_dest_node = random.choice(self.ws_nodes)  # since the vehicle is empty, assign one workstation
            if new_agv_pos[0] != -999:
                #print(agv_number_max, 'agv_new', new_agv_pos, ori_node, new_agv_dest_node)
                new_agv = AGV(ori_node, new_agv_dest_node, agv_number_max,0)
                new_agv.path = nx.shortest_path(self.graph, ori_node, new_agv_dest_node, weight='dist')
                self.agv_fleet.append(new_agv)
                agv_number_max += 1
        # start simulation
        for t in range(maxT):
            #print(t)
            #self.agv_fleet[0].print_s()
            #print('loc',self.node2loc(self.agv_fleet[0].current_node))
            #self.agv_fleet[2].print_s()
            self.deadlock_graph = nx.DiGraph() #deadlock graph record which agv is stucked by which, if there's circle, then there's a deadlock
            for agv in self.agv_fleet:
                self.deadlock_graph.add_node(agv.unique_index) # initialize graph
            if time.time()-self.caltime0>100:
                print('time out, blocked at t=', t, 'maxT:', maxT, 'number AGV:',num_AGV)
                break
            speed = 0
            # simulation steps:
            # step 1. set tasks for agvs if countdown=0 and arrived at dest (become idle)
            # step 2. plan next moves and collision detect
            # step 3. countdown and execute

            # step 1. set new tasks for agvs
            complete_agv = []
            for agv in self.agv_fleet:
                if agv.check_task_complete():
                    complete_agv.append(agv)
            for agv in complete_agv: # plan for arrived agv
                assert agv.countdown>=0
                #agv.print_s()
                if (agv.origin_node,agv.dest_node) in self.od_travel_time.keys():
                    self.od_travel_time[(agv.origin_node,agv.dest_node)].append(
                        t - agv.current_task_start_time
                    )
                else:
                    self.od_travel_time[(agv.origin_node, agv.dest_node)]=[
                        t - agv.current_task_start_time
                    ]
                    self.od_travel_time_theory[(agv.origin_node, agv.dest_node)] = [0,0]
                if agv.load == 0: # if just arrive at workstation
                    agv.load_item(load_time=Tload)
                    agv.path = [agv.current_node]
                elif agv.load == 1: # if just arrive at dropoff point
                    agv.unload_item(unload_time=Tdrop)
                    fulfilled += 1
                    agv.path = [agv.current_node]
                    self.dropoff_map[self.node2loc(agv.current_node)] += 1
                elif agv.load == 2 and agv.countdown==0: # if finish loading, to dropoff
                    if use_sp:
                        agv.load = 1
                        agv.assigne_new_dest(self.find_feasible_position_initial(*random.choice(self.dropoff_blocks)), t, True)
                    else:
                        agv.load = 1
                        dest_set = set([self.find_feasible_position_initial(*block) for block in self.dropoff_blocks])
                        new_agv_dest, new_agv_path = self.follow_path(agv.current_node, dest_set, 'B')
                        agv.assigne_new_dest(new_agv_dest, t, True)
                        agv.path = new_agv_path
                elif agv.load == 3 and agv.countdown==0: # if finish unloading, to workstation
                    if use_sp:
                        agv.load = 0
                        agv.assigne_new_dest(random.choice(self.ws_nodes), t, True)
                    else:
                        agv.load = 0
                        dest_set = set(self.ws_nodes)
                        new_agv_dest, new_agv_path = self.follow_path(agv.current_node, dest_set, 'A')
                        agv.assigne_new_dest(new_agv_dest,t, True)
                        agv.path = new_agv_path
                if use_sp:
                    agv.path = nx.shortest_path(self.graph, agv.current_node, agv.dest_node, weight='dist')
                #self.agv_paths[agv_i] = nx.shortest_path(self.graph, self.agv_nodes[agv_i], self.agv_dest[agv_i])
            fulfill_record.append(fulfilled)

            # plan next nove for agvs, state_map is a map such that agv can claim blocks on it
            for agv in self.agv_fleet:
                if agv.countdown == 0 and agv.load < 2: # if task is finished, try to access next node and update countdown
                    agv.next_node = agv.path[1]
                    agv.countdown = int(self.graph.edges[agv.current_node, agv.next_node]['dist'])
            # check conflicts
            self.state_map = np.zeros((n,m))  # state_map=0 if no conflict, otherwise potential conflict
            self.physical_state_map = np.zeros((n,m))
            for agv in self.agv_fleet:
                loc_this = self.node2loc(agv.current_node)
                if loc_this[0]!=-999:
                    self.state_map[loc_this[0], loc_this[1]] = agv.unique_index+1
                    self.physical_state_map[loc_this[0],loc_this[1]] = agv.unique_index+1
                loc_prev = self.node2loc(agv.previous_node)
                if loc_prev[0]!=-999:
                    self.state_map[loc_prev[0], loc_prev[1]] = agv.unique_index+1
            # check conflict
            for agv in self.agv_fleet:
                loc_this = self.node2loc(agv.current_node)
                loc_next = self.node2loc(agv.next_node)
                if loc_this != loc_next: # if plan to move
                    if (loc_next[0]!=-999) and (self.state_map[loc_next[0], loc_next[1]] >= 1): # if the next is occupied
                        self.deadlock_graph.add_edge(agv.unique_index, self.state_map[loc_next[0], loc_next[1]]-1) # construct deadlock detection graph
                        agv.next_node = agv.current_node # abandon movement and stay
                        total_delay += 1  # if conflict, stay and wait for 1 time unit
                        #if agv.unique_index==2:
                        #    print('agv2 blocked')
                        agv.delay_time += 1
                        self.block_flow_table[(loc_next[0], loc_next[1])][2] += 1
                    elif loc_next[0]!=-999:
                        self.state_map[loc_next[0], loc_next[1]] = agv.unique_index+1 #if not occupied, then claim it
            # check degrees
            undir_deadlock_graph = self.deadlock_graph.to_undirected()
            connect_components = [undir_deadlock_graph.subgraph(c).copy() for c in nx.connected_components(undir_deadlock_graph)]
            max_degree_list = [max(dict(subgraph.degree).values()) for subgraph in connect_components if max(dict(subgraph.degree).values())>0]

            total_num_queues += len(max_degree_list)
            total_num_complicated_queues += len([deg for deg in max_degree_list if deg>=3])
            total_num_simple_blocking += len([deg for deg in max_degree_list if deg <= 1])
            # check deadlock
            try:
                cycle = list(nx.find_cycle(self.deadlock_graph))
                if len(cycle)>0:
                    #print('deadlock found at time=',t)
                    key_agv = cycle[0][0]
                    self.agv_fleet[key_agv].move_to_dest()
                    fulfilled -= 1
                    deadlock_flag += 1
            except:
                #print('no deadlock')
                pass
            self.block_map += self.state_map.clip(max=1.) #heatmap of occupancy

            # step 3. do operations, countdown and move
            #self.agv_fleet[8].print_s()
            for agv in self.agv_fleet:
                agv.previous_node = agv.current_node
            for agv in self.agv_fleet:
                agv.countdown -= 1
                if agv.countdown == 0:
                    loc_this = self.node2loc(agv.current_node)
                    loc_next = self.node2loc(agv.next_node)
                    if loc_this!=loc_next:
                        speed+=1
                    if agv.current_node != agv.next_node: # if we need to move
                        self.graph[agv.current_node][agv.next_node]['flow'] += 1
                        agv.current_node = agv.next_node # then move
                        agv.path.pop(0) # remove the first one
                        assert agv.current_node == agv.path[0]
                        if loc_this[0]!=-999:
                            if loc_next == loc_this:
                                self.block_flow_table[(loc_this[0], loc_this[1])][0] += 1 #0: turning flow
                            else:
                                self.block_flow_table[(loc_this[0], loc_this[1])][1] += 1 #1: total flow

            existing_agv = len(self.agv_fleet)
            if existing_agv > 0:
                avg_speed = speed / existing_agv
            else:
                avg_speed = 0
            active_agv_number_record.append(existing_agv)
            speed_record.append(avg_speed)
        total_delay_theory, total_delay_det, total_delay_simp,_,block_rate = self.cal_theory_delay(maxT=maxT, num_AGV=num_AGV)

        f = open("out.txt", 'a')
        print(num_AGV, max(fulfill_record), np.mean(speed_record), total_delay, total_delay_theory, total_delay_det, total_delay_simp, np.max(block_rate), file=f)
        if total_num_queues>=1:
            print('chance of one-one blocking', total_num_simple_blocking/total_num_queues,file=f)
            print('chance of non-independent queue', total_num_complicated_queues/total_num_queues, file=f)
            print('rate of non-independent queue', total_num_complicated_queues/MAX_T, file=f)
            print('block rate max', np.max(block_rate), file=f)
        f.close()
        if total_num_queues>0:
            return speed_record, active_agv_number_record, fulfill_record, total_delay, total_delay_theory, total_delay_det, total_delay_simp, self.block_flow_table, self.block_map, deadlock_flag,\
                   total_num_simple_blocking / total_num_queues, total_num_complicated_queues/total_num_queues,  total_num_complicated_queues/MAX_T
        else:
            return speed_record, active_agv_number_record, fulfill_record, total_delay, total_delay_theory, total_delay_det, total_delay_simp, self.block_flow_table, self.block_map, deadlock_flag,\
                   0, 0, 0
    def cal_cell_flow(self, i,j):
        flow_total = 0
        flow_turn = 0
        flow_drop = self.dropoff_map[i, j]
        block_nodes = [4 * (self.M * i + j) + k for k in range(4)]
        list_inedges = []
        for node in block_nodes:  # calculate flow at one block
            in_edges = self.graph.in_edges(node)
            for edge in in_edges:
                if edge[0] not in block_nodes:
                    flow_total += self.graph[edge[0]][edge[1]]['flow']
                    list_inedges.append((edge[0], edge[1]))
                else:
                    flow_turn += self.graph[edge[0]][edge[1]]['flow']
        return flow_total, flow_turn, flow_drop, list_inedges
    def cal_theory_delay(self, maxT=1, num_AGV=10, get_block_rate=True):
        deadlock_flag = 0
        # calculate theoretical delay
        m, n = self.M, self.N
        self.blocking_prob = np.zeros((n,m))
        #self.block_flow_table_theory record block flow and delay calculated
        self.block_flow_table_theory = {(i,j):[0,0,0,0,0] for i in range(n) for j in range(m)} #['flow_turn','flow_total','delay_simp','delay_mg1","flow_drop"]
        total_delay = 0
        total_delay_det = 0
        total_delay_simp = 0
        for i in range(n):
            for j in range(m):
                flow_total, flow_turn, flow_drop,list_inedges = self.cal_cell_flow(i,j)
                self.block_flow_table_theory[(i,j)][0] = flow_turn
                self.block_flow_table_theory[(i,j)][1] = flow_total
                self.block_flow_table_theory[(i, j)][4] = flow_drop

                flow_drop = flow_drop / maxT * (num_AGV-1)/num_AGV
                flow_turn = flow_turn / maxT * (num_AGV-1)/num_AGV
                flow_total = flow_total / maxT * (num_AGV-1)/num_AGV
                if flow_drop+flow_turn>flow_total:
                    flow_total = flow_drop+flow_turn

                if flow_total == 0:
                    delay_block = 0
                    delay_block_simp = 0
                else:
                    mu_inv = 2 + Tt * flow_turn / flow_total + Tdrop *flow_drop/flow_total
                    if 2 - 4 * flow_total - 2 * flow_turn * Tt - 2*Tdrop*flow_drop< 0:
                        warnings.warn('rho >1')
                        delay_block = 0
                        delay_block_simp = 0
                        deadlock_flag+=1
                    else:
                        delay_block = ((flow_total-flow_turn-flow_drop) * 4. + flow_turn * (2.+Tt) ** 2. + flow_drop*(2.+Tdrop)**2) / (
                                    2 - 4 * flow_total - 2 * flow_turn * Tt- 2*flow_drop*Tdrop)
                        delay_block = max(delay_block, 0)
                        delay_block_simp = 0.5*((flow_total-flow_turn-flow_drop) * 4. + flow_turn * (2.+Tt)** 2+flow_drop*(2.+Tdrop)**2) # delay based on renewal process
                self.block_flow_table_theory[(i, j)][2] = delay_block_simp
                self.block_flow_table_theory[(i, j)][3] = delay_block
                if num_AGV-1>0:
                    total_delay += delay_block * flow_total * maxT * num_AGV/(num_AGV-1)
                    total_delay_simp += delay_block_simp * flow_total * maxT*num_AGV/(num_AGV-1)
                for edge in list_inedges:
                    self.graph[edge[0]][edge[1]]['delay'] = delay_block
                # cal blocking probabilities
                if get_block_rate:
                    for p in range(n):
                        for q in range(m):
                            if (abs(p-i)+abs(q-i)) < 5 and (abs(p-i)+abs(q-i)>0):
                                if num_AGV>1:
                                    self.blocking_prob[i,j] += flow_total*num_AGV/(num_AGV-1)*self.cal_blocking_prob((p,q),(i,j))

        # calculate the 1st moment of block delay on all OD paths
        for key in self.od_travel_time_theory.keys():
            sp_od = nx.shortest_path(self.graph, key[0], key[1], weight='weight')
            sp_length = nx.shortest_path_length(self.graph, key[0], key[1], weight='weight')
            blocks = [self.node2loc(node) for node in sp_od]
            blocks = list(set(blocks))
            total_delay_simp_route = 0.
            total_delay_mg1_route = 0.
            total_delay_simp_var_route = 0.
            for block in blocks:
                if block[0]!=-999:
                    total_delay_mg1_route += self.block_flow_table_theory[(block[0], block[1])][3]
                    total_delay_simp_route += self.block_flow_table_theory[(block[0], block[1])][2]
                #total_delay_simp_var_route += self.block_flow_table_theory[(block[0], block[1])][4] - self.block_flow_table_theory[(block[0], block[1])][2]**2
            self.od_travel_time_theory[key] = [sp_length, total_delay_simp_route, total_delay_mg1_route]
        return total_delay, total_delay_det, total_delay_simp, deadlock_flag, self.blocking_prob

    def follow_path(self, ori, dest_set, type):
        # follow "flow_ij_A" (from dropoff to workstation) and "flow_ij_B" (from workstation to dropoff)
        current_node = ori
        path = [current_node]
        while current_node not in dest_set:
            next_node_list = [edge[1] for edge in self.graph.out_edges(current_node)]
            next_node_weight = [abs(self.graph.edges[(edge[0],edge[1])]['flow_ij_'+type]) for edge in self.graph.out_edges(current_node)]
            if sum(next_node_weight)<1e-10:
                print('error, uncharted location')
                next_node_weight = [1. for edge in self.graph.out_edges(current_node)]
            random_choice = random.choices(next_node_list,weights=next_node_weight,k=1)
            random_choice = random_choice[0]
            path.append(random_choice)
            current_node = random_choice
        return current_node, path

    def calc_sp_assignment(self, od_flow = 0.1, maxT = 5000, num_AGV=10, get_block_rate=True):
        # assign flow according to shortest path
        self.dropoff_map = np.zeros((self.N, self.M))
        for dest in self.dropoff_blocks:
            self.dropoff_map[dest] = od_flow*maxT

        for ori in self.ws_nodes:
            for dest in self.dropoff_blocks:
                start_node = ori
                end_node = self.find_feasible_position_initial(*dest)
                self.od_travel_time_theory[(start_node,end_node)] = [0,0]
                sp = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
                for i in range(len(sp)-1):
                    self.graph[sp[i]][sp[i+1]]['flow'] += od_flow*maxT
        for ori in self.dropoff_blocks:
            for dest in self.ws_nodes:
                start_node = self.find_feasible_position_initial(*ori)
                end_node = dest
                self.od_travel_time_theory[(start_node, end_node)] = [0, 0]
                sp = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
                for i in range(len(sp)-1):
                    self.graph[sp[i]][sp[i+1]]['flow'] += od_flow*maxT
        # calculate delay accordingly
        total_delay_theory, total_delay_det, total_delay_simp, deadlock_flag, block_rate = self.cal_theory_delay(maxT=maxT, num_AGV=num_AGV,get_block_rate=get_block_rate)
        return self.block_flow_table_theory,  total_delay_theory, total_delay_det, total_delay_simp, deadlock_flag, block_rate
    
    def cal_blocking_prob(self, cell1, cell2):
        # cell1 and cell2 is in (i,j)
        nodes1 = self.cell2nodes(*cell1)
        nodes2 = self.cell2nodes(*cell2)
        # find shortest path from cell 2 to cell 1, with length
        sp_length = BIG_NUM
        sp_12 = []
        for node1 in nodes1:
            for node2 in nodes2:
                sp = nx.shortest_path(self.graph, node2, node1)
                if len(list(sp))<sp_length:
                    n1 = node1
                    n2 = node2
                    sp_12 = sp
                    sp_length = len(sp)

        if cell1[0]!=cell2[0] and cell1[1]!=cell2[1]:
            sp_length-=1
        crit_flow = BIG_NUM
        for i in range(len(sp_12)-1):
            if self.graph[sp_12[i]][sp_12[i+1]]['flow']<crit_flow:
                crit_flow = self.graph[sp_12[i]][sp_12[i+1]]['flow']
        if sp_length>5 or crit_flow<1e-5:
            return 0
        flow_total, flow_turn, flow_drop, _ = self.cal_cell_flow(cell1[0],cell1[1])
        f1 = (flow_total-flow_turn-flow_drop)/MAX_T
        f2 = flow_turn/MAX_T
        f3 = flow_drop/MAX_T
        pmf_cell1 = self.cdf_first_3(cell1[0],cell1[1],f1, f2, f3)
        prob_block = 0
        for n in range(sp_length, 4):
            if f1+f2+f3>1e-10:
                prob_block += pmf_cell1[n]*(1-binom.cdf(sp_length,n,crit_flow/MAX_T/(f1+f2+f3)))
        # for prob(queue from cell 1 = n), (n> length), prob(blocking cell2) =  prob(queue from cell 1 = n) * p_binom(f_critcal/f_cell1, n: X>= length)
        if math.isnan(prob_block):
            return 0
        return prob_block

def new_workstation_param(setting='A'):
    if setting=='A':
        workstation_enter_blocks = [(6, 0), (12, 0)]
        workstation_exit_blocks = [(7,0),(13,0)]
        noaccess_blocks = [(2 + 3 * i, 4 + 3 * j) for i in range(6) for j in range(5)]
        dropoff_position = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dropoff_blocks = [(block[0] + div[0], block[1] + div[1]) for block in noaccess_blocks for div in dropoff_position]
        allow_up_col = [0, 2, 5, 8, 11, 14, 17]
        allow_down_col = [1, 3, 6, 9, 12, 15, 18]
        allow_left_row = [0, 3, 6, 9, 12, 15, 18]
        allow_right_row = [1, 4, 7, 10, 13, 16, 19]
        m,n = 19,20
    elif setting=='B':
        #workstation_blocks = [(5, 0), (14, 0)]
        workstation_enter_blocks = [(6, 0), (12, 0)]
        workstation_exit_blocks = [(7, 0), (13, 0)]
        noaccess_blocks = [(2 + 3 * i, 4 + 3 * j) for i in range(6) for j in range(3)]
        dropoff_position = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dropoff_blocks = [(block[0] + div[0], block[1] + div[1]) for block in noaccess_blocks for div in
                          dropoff_position]
        allow_up_col = [0, 2, 5, 8, 11]
        allow_down_col = [1, 3, 6, 9, 12]
        allow_left_row = [0, 3, 6, 9, 12, 15, 18]
        allow_right_row = [1, 4, 7, 10, 13, 16, 19]
        m, n = 13, 20
    elif setting=='C':
        workstation_enter_blocks = [(3,0), (6, 0), (12, 0),(15,0)]
        workstation_exit_blocks = [(4,0), (7, 0), (13, 0),(16,0)]
        noaccess_blocks = [(2 + 3 * i, 4 + 3 * j) for i in range(6) for j in range(5)]
        dropoff_position = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dropoff_blocks = [(block[0] + div[0], block[1] + div[1]) for block in noaccess_blocks for div in
                          dropoff_position]
        allow_up_col = [0, 2, 5, 8, 11, 14, 17]
        allow_down_col = [1, 3, 6, 9, 12, 15, 18]
        allow_left_row = [0, 3, 6, 9, 12, 15, 18]
        allow_right_row = [1, 4, 7, 10, 13, 16, 19]
        m, n = 19, 20

    return m,n,workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row



if __name__ =='__main__':
    #warehouse = WarehouseMap(10,10)
    flow = 0.5
    #speed_record, active_agv_number_record, fulfill_record, total_delay,total_delay_theory,total_delay_theory2,total_delay_det, block_flow, block_occ= warehouse.run_simulation(input_flow=flow,maxT=MAX_T)
    m,n, workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row = new_workstation_param('A')
    warehouse = WarehouseMap(m,n,workstation_enter_blocks, workstation_exit_blocks, dropoff_blocks, noaccess_blocks, allow_up_col, allow_down_col, allow_left_row, allow_right_row)
    # #warehouse.so_gurobi(0.0007)
    # #warehouse.read_optimal_flow()
    speed_record, active_agv_number_record, fulfill_record, total_delay,total_delay_theory,total_delay_theory2,total_delay_det, block_flow, block_occ, deadlock_flag,_,_,_= \
        warehouse.run_simulation(num_AGV=12, maxT=MAX_T, use_sp=True)
    #
    # #
    # import pandas as pd
    # df_graph = pd.DataFrame.from_dict(dict(warehouse.graph.edges),orient='index')
    # df_graph.to_csv('temp1.csv')
    # df_block = pd.DataFrame.from_dict(block_flow,orient='index',columns=['flow_turn','flow_straight','delay'])
    # block_occ_dict = {(i,j):[block_occ[i,j]] for i in range(10) for j in range(10)}
    # df_block_occ =  pd.DataFrame.from_dict(block_occ_dict,orient='index',columns=['occ'])
    # df_block_join = df_block.join(df_block_occ)
    # df_block_theory = pd.DataFrame.from_dict(warehouse.block_flow_table_theory,orient='index',columns=['flow_turn_the','total_flow_the','delay_simp','delay_q','flow_drop'])
    # df_block_join = df_block_join.join(df_block_theory)
    # df_block_join.index = pd.MultiIndex.from_tuples(df_block_join.index)
    # df_block_join['rho'] = 2*df_block_join['total_flow_the']/MAX_T+Tt*df_block_join['flow_turn_the']/MAX_T+Tdrop*df_block_join['flow_drop']/MAX_T
    # df_block_join.to_csv('temp_block1.csv')
    # avg_var = {key:[np.mean(item), np.var(item)] for key, item in warehouse.od_travel_time.items()}



# draw



#
# from multiprocessing import Pool, cpu_count
# if __name__ == '__main__':
#     exp_num = 100
#     agv_list = [max(1,int(i/3)) for i in range(1,exp_num+1)]
#     #agv_list = [10]
#     with Pool(cpu_count()) as p:
#         results = p.map(return_speed, agv_list)
#
#     # with Pool(cpu_count()) as p:
#     #     results_so = p.map(return_speed_so, agv_list)
#
#     import matplotlib.pyplot as plt
#
#     results = np.array(results)
#    #  plt.figure(0)
#    #  plt.scatter(results[:, 1] / MAX_T, results[:, 0])
#    #  plt.xlabel('output flow')
#    #  plt.ylabel('avg speed')
#    #  plt.savefig('agv-speed.png')
#    #
#    #  plt.figure(1)
#    #  plt.scatter(agv_list, results[:, 0])
#    #  plt.xlabel('number of agv')
#    #  plt.ylabel('avg speed')
#    #
#     plt.figure(2)
#     plt.scatter(agv_list, results[:, 1])
#     plt.xlabel('number of agv')
#     plt.ylabel('output flow')
#    #  plt.savefig('input_output.png')
#    #
#     plt.figure(3)
#     plt.scatter(agv_list, results[:, 3], label='total delay simulation')
#     plt.scatter(agv_list, results[:, 4], label='total delay queue')
#    # plt.scatter(flow_list, results[:, 5], label='total delay QNA')
#     #plt.scatter(flow_list, results[:, 5], label='total delay det')
#     plt.scatter(agv_list, results[:, 6], label='total delay simple')
#     plt.legend()
#     plt.xlabel('number of agv')
#     plt.ylabel('total delay')
#    #  plt.savefig('input_delay.png')
#    #
#     plt.figure(4)
#     plt.scatter(results[:, 1] / MAX_T, results[:, 3], label='total delay simulation')
#     plt.scatter(results[:, 1] / MAX_T, results[:, 4], label='total delay queue')
#     #plt.scatter(results[:, 1] / MAX_T, results[:, 5], label='total delay QNA')
#     #plt.scatter(results[:, 1] / MAX_T, results[:, 5], label='total delay det')
#     plt.scatter(results[:, 1] / MAX_T, results[:, 6], label='total delay simple')
#     plt.legend()
#     plt.xlabel('output flow')
#     plt.ylabel('total delay')
#     plt.savefig('output_delay.png')

    # import pandas as pd
    #
    # results_all = [results[i]+results_so[i] for i in range(len(results))]
    # results_df = pd.DataFrame(results_all, columns=['avg_speed','outflow_number','max_number_of_AGV_at_same_time','delay_simulation',
    #                                             'delay_MG1','delay_DG1','delay_arrival_process',
    #                                             'avg_speed_so', 'outflow_number_so', 'max_number_of_AGV_at_same_time_so',
    #                                             'delay_simulation_so','delay_MG1_so', 'delay_DG1_so', 'delay_arrival_process_so',
    #                                             ])
    # results_df['number of AGV'] = 0.
    # results_df['number of AGV'] = np.array(agv_list)
    # results_df['max_T'] = MAX_T
    #
    # results_df['TH: shortest path'] = results_df['outflow_number']/5000
    # results_df['TH: SO routing+assignment'] = results_df['outflow_number_so'] / 5000
    # results_df.to_csv('result_exp2.csv')



