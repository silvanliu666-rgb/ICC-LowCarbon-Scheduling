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

P_MAX = 5000; T = 96; PUE_CONSTANT = 1.2
COST_PV = 0.20; COST_WIND = 0.15; COST_DR = 30.0


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
    PV = df_res['PV'].values; Wind = df_res['Wind'].values
    df_wl = pd.read_csv('original_workload.csv')
    LNM = df_wl['LNM_kW'].values
    LMS_ori = df_wl['LMS_kW'].values
    return LNM, LMS_ori, Price, PV, Wind


class SchedulerEconomicOnly:
    def __init__(self, LNM, LMS_ori, Price, PV, Wind):
        self.LNM = LNM; self.LMS_ori = LMS_ori
        self.Price = Price; self.PV = PV; self.Wind = Wind
        self.tasks_LMC = [{'p': 1008.11, 'dur': 8}, {'p': 504.05, 'dur': 16}]
        self.tasks_LMI = [{'p': 403.24, 'dur': 20}, {'p': 336.04, 'dur': 16}]
        self.pop_size = 60; self.max_iter = 100

    def calculate_fitness(self, ind):
        LMS_new = (np.array(ind['lms']) / (np.sum(ind['lms']) + 1e-6)) * np.sum(self.LMS_ori)
        LMC_new = np.zeros(T); LMI_new = np.zeros(T); cost_dr = 0.0
        for i, t in enumerate(self.tasks_LMC):
            s = ind['lmc'][i]; LMC_new[s:s+t['dur']] += t['p']
        for i, t in enumerate(self.tasks_LMI):
            active = np.argsort(np.array(ind['lmi'][i]))[-t['dur']:]
            LMI_new[active] += t['p']
            u = np.zeros(T); u[active] = 1
            cost_dr += np.sum(np.abs(np.diff(u))) * COST_DR
            
        P_total = (self.LNM + LMS_new + LMC_new + LMI_new) * PUE_CONSTANT
        P_net = np.maximum(0, P_total - self.PV - self.Wind)
        
     
        cost_grid = np.sum(P_net * self.Price * 0.25)
        cost_op = np.sum(self.PV) * COST_PV * 0.25 + np.sum(self.Wind) * COST_WIND * 0.25
        em_real = np.sum(P_net * E_GRID_dynamic * 0.25)
        
        cost_op = np.sum(self.PV) * COST_PV * 0.25 + np.sum(self.Wind) * COST_WIND * 0.25
        em_real = np.sum(P_net * E_GRID_dynamic * 0.25)
        
    
        penalty = 1e7 if np.max(P_total) > P_MAX * 1.05 else 0
        
        fitness = cost_grid + cost_op + cost_dr + penalty
        return fitness, fitness, em_real

    def run(self):
        pop = [{'lms': [random.random() for _ in range(T)], 
                'lmc': [random.randint(0, T-t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI]} for _ in range(self.pop_size)]
        best_ind = None; min_fit = float('inf')
        for it in range(self.max_iter):
            fits = [self.calculate_fitness(ind)[0] for ind in pop]
            if min(fits) < min_fit: min_fit = min(fits); best_ind = copy.deepcopy(pop[np.argmin(fits)])
            new_pop = [pop[i] for i in np.argsort(fits)[:self.pop_size//2]]
            while len(new_pop) < self.pop_size:
                c = copy.deepcopy(random.choice(new_pop[:10]))
                c['lmc'] = [random.randint(0, T-t['dur']) for t in self.tasks_LMC]
                new_pop.append(c)
            pop = new_pop
        return best_ind


def calculate_baseline_aligned(LNM, LMS_ori, PV, Wind, Price):
    LMC_base = np.zeros(T); LMI_base = np.zeros(T)
    LMC_base[72:72+8] += 1008.11; LMC_base[72:72+16] += 504.05
    LMI_base[72:72+20] += 403.24; LMI_base[72:72+16] += 336.04
    
    P_total_base = (LNM + LMS_ori + LMC_base + LMI_base) * PUE_CONSTANT
    P_net_base = np.maximum(0, P_total_base - PV - Wind)
    
    cost_grid_base = np.sum(P_net_base * Price * 0.25)
    cost_op_base = np.sum(PV) * COST_PV * 0.25 + np.sum(Wind) * COST_WIND * 0.25
    em_base = np.sum(P_net_base * E_GRID_dynamic * 0.25)
    
    return (cost_grid_base + cost_op_base), em_base


if __name__ == "__main__":
    LNM, LMS_ori, Price, PV_base, Wind_base = load_data()
  
    total_dc = (np.sum(LNM)+np.sum(LMS_ori)+16128+20496)*PUE_CONSTANT 
    base_ratio = (np.sum(PV_base)+np.sum(Wind_base))/total_dc
    
    target_ratios = np.arange(0.0, 1.01, 0.1); res_x = []; costs = []; ems = []
    
    print("Running sensitivity analysis...")
    for r in target_ratios:
        scale = r / base_ratio if base_ratio > 0 else 0
        PV, Wind = PV_base*scale, Wind_base*scale
        
        cost_b, em_b = calculate_baseline_aligned(LNM, LMS_ori, PV, Wind, Price)
        scheduler = SchedulerEconomicOnly(LNM, LMS_ori, Price, PV, Wind)
        best = scheduler.run()
        _, cost_o, em_o = scheduler.calculate_fitness(best)
        
        c_red = (cost_b - cost_o)/cost_b*100; e_red = (em_b - em_o)/em_b*100
        res_x.append(r*100); costs.append(c_red); ems.append(e_red)
        print(f"Rate {r*100:3.0f}% -> Cost Reduction: {c_red:5.2f}%, Carbon Emission Reduction: {e_red:5.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(res_x, costs, marker='s', color='#d62728', linewidth=2, label='Cost Reduction Rate (%)')
    plt.plot(res_x, ems, marker='o', color='#2ca02c', linewidth=2, label='Carbon Emission Reduction Rate (%)')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Renewable Energy Percentage (%)', fontweight='bold', fontsize=12)
    plt.ylabel('Reduction Rate Compared to Baseline (%)', fontweight='bold', fontsize=12)
    plt.xticks(np.arange(0, 101, 10))
    plt.legend(loc='best', frameon=True, fontsize=15, framealpha=0.9)
    plt.grid(False)
    plt.savefig('Fig_Paradox_Final_Account.png', dpi=300, bbox_inches='tight')
    print("\nALL DONE")