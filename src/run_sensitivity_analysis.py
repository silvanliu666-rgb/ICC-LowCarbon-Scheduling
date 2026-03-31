import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

P_MAX = 5000; T = 96; C_QUOTA = 20000
PRICE_CARBON = 0.090; PRICE_GEC = 0.005; CARBON_MU = 0.25; CARBON_L = 2000       
PUE_CONSTANT = 1.2; COST_PV = 0.20; COST_WIND = 0.15; COST_DR = 30.0

E_GRID_AVG = 0.5703 


E_GRID_dynamic = np.zeros(T)
for t in range(T):
    hour = t * 0.25
    if hour < 8.0 or hour >= 23.0: 
        E_GRID_dynamic[t] = 0.85 
    else: 
        E_GRID_dynamic[t] = 0.35

def load_data():
    Price = pd.read_csv('TOU.csv')['Price'].values
    df_res = pd.read_csv('res_data.csv')
    PV = df_res['PV'].values
    Wind = df_res['Wind'].values
    df_wl = pd.read_csv('original_workload.csv')
    LNM = df_wl['LNM_kW'].values
    LMS_ori = df_wl['LMS_kW'].values
    return LNM, LMS_ori, Price, PV, Wind


class SchedulerMOO:
    def __init__(self, LNM, LMS_ori, Price, PV, Wind, mode='weighted'):
        self.LNM = LNM
        self.LMS_ori = LMS_ori
        self.Price = Price
        self.PV = PV
        self.Wind = Wind
        
        self.tasks_LMC = [
            {'id': 1, 'p': 1008.11, 'dur': 8,  'e': 0, 'l': 96},
            {'id': 2, 'p': 504.05,  'dur': 16, 'e': 0, 'l': 96}
        ]
        self.tasks_LMI = [
            {'id': 1, 'p': 403.24, 'dur': 20},
            {'id': 2, 'p': 336.04, 'dur': 16}
        ]
        
    
        self.pop_size = 80   
        self.max_iter = 150
        
        self.extremes = {
            'F1_min': 1.0, 'F1_max': 1.0,
            'F2_min': 1.0, 'F2_max': 1.0
        }
        self.w1 = 0.5  
        self.w2 = 0.5  
        self.mode = mode 

    def calculate_fitness(self, ind):
        prop = np.array(ind['lms']) / (np.sum(ind['lms']) + 1e-6)
        LMS_new = prop * np.sum(self.LMS_ori)
        
        LMC_new = np.zeros(T)
        for i, t in enumerate(self.tasks_LMC):
            s = ind['lmc'][i]
            LMC_new[s:s+t['dur']] += t['p']
            
        LMI_new = np.zeros(T)
        cost_dr = 0.0  
        for i, t_task in enumerate(self.tasks_LMI):
            priorities = np.array(ind['lmi'][i])
            window = np.ones(4) 
            smoothed_priorities = np.convolve(priorities, window, mode='same')
            active_slots = np.argsort(smoothed_priorities)[-t_task['dur']:]
            LMI_new[active_slots] += t_task['p']
            
            u_status = np.zeros(T)
            u_status[active_slots] = 1
            switches = np.sum(np.abs(np.diff(u_status)))
            cost_dr += switches * COST_DR
            
        P_IT = self.LNM + LMS_new + LMC_new + LMI_new
        P_total = P_IT * PUE_CONSTANT
        
        P_net = np.maximum(0, P_total - self.PV - self.Wind)
        cost_grid = 0
        em_real = 0
        for t in range(T):
            if P_net[t] > 0:
                cost_grid += P_net[t] * self.Price[t] * 0.25
                em_real += P_net[t] * E_GRID_dynamic[t] * 0.25 
        
        net_em = max(0, em_real - C_QUOTA)
        cost_green = 0; cost_carbon = 0
        
        if net_em > 0:
            gec_unit_cost = PRICE_GEC / E_GRID_AVG 
            if gec_unit_cost < PRICE_CARBON:
                cost_green = net_em * gec_unit_cost
            else:
                if net_em <= CARBON_L: cost_carbon = net_em * PRICE_CARBON
                elif net_em <= 2 * CARBON_L: cost_carbon = CARBON_L * PRICE_CARBON + (net_em - CARBON_L) * PRICE_CARBON * (1 + CARBON_MU)
                else: cost_carbon = CARBON_L * PRICE_CARBON * (2 + CARBON_MU) + (net_em - 2 * CARBON_L) * PRICE_CARBON * (1 + 2 * CARBON_MU)
                
        cost_op = np.sum(self.PV) * COST_PV * 0.25 + np.sum(self.Wind) * COST_WIND * 0.25
        
        F1 = cost_grid + cost_op + cost_green + cost_carbon + cost_dr
        F2 = em_real 
        
        penalty = 1e7 if np.max(P_total) > P_MAX * 1.05 else 0
            
        if self.mode == 'min_F1': fitness = F1 + penalty
        elif self.mode == 'min_F2': fitness = F2 + penalty
        elif self.mode == 'weighted':
            norm_F1 = (F1 - self.extremes['F1_min']) / (self.extremes['F1_max'] - self.extremes['F1_min'] + 1e-6)
            norm_F2 = (F2 - self.extremes['F2_min']) / (self.extremes['F2_max'] - self.extremes['F2_min'] + 1e-6)
            fitness = self.w1 * norm_F1 + self.w2 * norm_F2 + penalty
            
        return fitness, F1, F2

    def run(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {
                'lms': [random.random() for _ in range(T)],
                'lmc': [random.randint(t['e'], t['l'] - t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
            }
            pop.append(ind)
            
        best_ind = None; min_fit = float('inf')
        
        for it in range(self.max_iter):
            fits = []
            for ind in pop:
                f, _, _ = self.calculate_fitness(ind)
                fits.append(f)
                if f < min_fit:
                    min_fit = f
                    best_ind = copy.deepcopy(ind)
            
            sorted_idx = np.argsort(fits)
            new_pop = [pop[i] for i in sorted_idx[:self.pop_size//2]]
            while len(new_pop) < self.pop_size:
                p = random.choice(new_pop[:10])
                c = copy.deepcopy(p)
                if random.random() < 0.7: c['lmc'] = [random.randint(t['e'], t['l'] - t['dur']) for t in self.tasks_LMC]
                if random.random() < 0.7: c['lms'][random.randint(0,T-1)] = random.random()
                if random.random() < 0.7: c['lmi'] = [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
                new_pop.append(c)
            pop = new_pop
            
        return best_ind


def calculate_baseline_aligned(LNM, LMS_ori, PV, Wind, Price):
    LMC_base = np.zeros(T); LMI_base = np.zeros(T)
   
    LMC_base[72:72+8] += 1008.11; LMC_base[72:72+16] += 504.05
    LMI_base[72:72+20] += 403.24; LMI_base[72:72+16] += 336.04
    
    P_total_base = (LNM + LMS_ori + LMC_base + LMI_base) * PUE_CONSTANT
    P_net_base = np.maximum(0, P_total_base - PV - Wind)
    
    cost_elec_base = 0; em_real_base = 0
    for t in range(T):
        if P_net_base[t] > 0:
            cost_elec_base += P_net_base[t] * Price[t] * 0.25
            em_real_base += P_net_base[t] * E_GRID_dynamic[t] * 0.25
            
    net_em_base = max(0, em_real_base - C_QUOTA)
    cost_green_base = 0; cost_carbon_base = 0
    if net_em_base > 0:
        gec_unit_cost = PRICE_GEC / E_GRID_AVG 
        if gec_unit_cost < PRICE_CARBON: 
            cost_green_base = net_em_base * gec_unit_cost
        else:
            if net_em_base <= CARBON_L: cost_carbon_base = net_em_base * PRICE_CARBON
            elif net_em_base <= 2 * CARBON_L: cost_carbon_base = CARBON_L * PRICE_CARBON + (net_em_base - CARBON_L) * PRICE_CARBON * (1 + CARBON_MU)
            else: cost_carbon_base = CARBON_L * PRICE_CARBON * (2 + CARBON_MU) + (net_em_base - 2 * CARBON_L) * PRICE_CARBON * (1 + 2 * CARBON_MU)
                
    cost_op_base = np.sum(PV)*COST_PV*0.25 + np.sum(Wind)*COST_WIND*0.25
    
    F1_base = cost_elec_base + cost_op_base + cost_green_base + cost_carbon_base
    F2_base = em_real_base 
    
    return F1_base, F2_base


if __name__ == "__main__":
    LNM, LMS_ori, Price, PV_base, Wind_base = load_data()
    
    total_it_power = np.sum(LNM) + np.sum(LMS_ori) + (1008.11*8 + 504.05*16) + (403.24*20 + 336.04*16)
    total_dc_power = total_it_power * PUE_CONSTANT
    base_res_power = np.sum(PV_base) + np.sum(Wind_base)
    base_ratio = base_res_power / total_dc_power
    
    target_ratios = np.arange(0.0, 1.01, 0.1)
    
    x_axis = []
    cost_reduction_rates = []
    em_reduction_rates = []
    
    print(f"Oprating Sensitivity Analysis")
    
    for r in target_ratios:
        print(f"-> Rate: {r*100:3.0f}% ...", end=" ")
        scale_factor = r / base_ratio if base_ratio > 0 else 0
        PV_scaled = PV_base * scale_factor
        Wind_scaled = Wind_base * scale_factor
        
      
        F1_ori, F2_ori = calculate_baseline_aligned(LNM, LMS_ori, PV_scaled, Wind_scaled, Price)
        
  
        scheduler_F1 = SchedulerMOO(LNM, LMS_ori, Price, PV_scaled, Wind_scaled, mode='min_F1')
        best_F1 = scheduler_F1.run()
        _, F1_min, F2_max = scheduler_F1.calculate_fitness(best_F1)
        
        scheduler_F2 = SchedulerMOO(LNM, LMS_ori, Price, PV_scaled, Wind_scaled, mode='min_F2')
        best_F2 = scheduler_F2.run()
        _, F1_max, F2_min = scheduler_F2.calculate_fitness(best_F2)
        
        scheduler_Final = SchedulerMOO(LNM, LMS_ori, Price, PV_scaled, Wind_scaled, mode='weighted')
        scheduler_Final.extremes = {'F1_min': F1_min, 'F1_max': F1_max, 'F2_min': F2_min, 'F2_max': F2_max}
        scheduler_Final.w1 = 0.5; scheduler_Final.w2 = 0.5
        
        best_ind_Final = scheduler_Final.run()
        _, final_F1, final_F2 = scheduler_Final.calculate_fitness(best_ind_Final)
        
  
        cost_reduce = (F1_ori - final_F1) / F1_ori * 100 if F1_ori > 0 else 0
        em_reduce = (F2_ori - final_F2) / F2_ori * 100 if F2_ori > 0 else 0
        
        x_axis.append(r * 100)
        cost_reduction_rates.append(cost_reduce)
        em_reduction_rates.append(em_reduce)
        
        print(f"✓ F1 Reduction: {cost_reduce:5.2f}%, F2 Reduction: {em_reduce:5.2f}%")


    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, cost_reduction_rates, marker='s', markersize=8, linestyle='-', color='#d62728', linewidth=2.5, label='Cost Reduction Rate (%)')
    plt.plot(x_axis, em_reduction_rates, marker='o', markersize=8, linestyle='-', color='#2ca02c', linewidth=2.5, label='Carbon Emission Reduction Rate (%)')
    
    plt.xlabel('Renewable Energy Percentage (%)', fontweight='bold', fontsize=12)
    plt.ylabel('Reduction Rate Compared to Baseline (%)', fontweight='bold', fontsize=12)
    plt.xticks(np.arange(0, 101, 10))
    plt.legend(loc='best', frameon=True, fontsize=15, framealpha=0.9)
    plt.grid(False)
    
    plt.savefig('Fig6_Sensitivity_MOO_Aligned.png', dpi=300, bbox_inches='tight')
    print("\nALL DONE")