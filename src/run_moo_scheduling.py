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

P_MAX = 5000         # kW
T = 96               # 15min/step
C_QUOTA = 20000      # kg 
PRICE_CARBON = 0.090 # CNY/kg
PRICE_GEC = 0.005    # CNY/kWh
CARBON_MU = 0.25      
CARBON_L = 2000       
PUE_CONSTANT = 1.2    
COST_PV = 0.20        
COST_WIND = 0.15      
COST_DR = 30.0        

E_GRID_AVG = 0.5703 


E_GRID_DYNAMIC = np.zeros(T)
for t in range(T):
    hour = t * 0.25
    if hour < 8.0 or hour >= 23.0: 
        E_GRID_DYNAMIC[t] = 0.85  
    else: 
        E_GRID_DYNAMIC[t] = 0.35  


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
        
        self.tasks_LMC = [{'id': 1, 'p': 1008.11, 'dur': 8}, {'id': 2, 'p': 504.05, 'dur': 16}]
        self.tasks_LMI = [{'id': 1, 'p': 403.24, 'dur': 20}, {'id': 2, 'p': 336.04, 'dur': 16}]
        
        self.pop_size = 80   
        self.max_iter = 150  
        
        self.extremes = {'F1_min': 1.0, 'F1_max': 1.0, 'F2_min': 1.0, 'F2_max': 1.0}
        self.w1 = 0.5; self.w2 = 0.5  
        self.mode = mode 

    def calculate_fitness(self, ind):
        LMS_new = (np.array(ind['lms']) / (np.sum(ind['lms']) + 1e-6)) * np.sum(self.LMS_ori)
        LMC_new = np.zeros(T); LMI_new = np.zeros(T); cost_dr = 0.0  
        
        for i, t in enumerate(self.tasks_LMC):
            s = ind['lmc'][i]; LMC_new[s:s+t['dur']] += t['p']
            
        for i, t_task in enumerate(self.tasks_LMI):
            priorities = np.array(ind['lmi'][i])
            window = np.ones(4) 
            smoothed = np.convolve(priorities, window, mode='same')
            active_slots = np.argsort(smoothed)[-t_task['dur']:]
            LMI_new[active_slots] += t_task['p']
            
            u_status = np.zeros(T); u_status[active_slots] = 1
            cost_dr += np.sum(np.abs(np.diff(u_status))) * COST_DR
            
        P_IT = self.LNM + LMS_new + LMC_new + LMI_new
        P_total = P_IT * PUE_CONSTANT
        P_net = np.maximum(0, P_total - self.PV - self.Wind)
        
        cost_grid = 0; em_real = 0
        for t in range(T):
            if P_net[t] > 0:
                cost_grid += P_net[t] * self.Price[t] * 0.25
                em_real += P_net[t] * E_GRID_DYNAMIC[t] * 0.25 
        
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
            
        return fitness, F1, F2, P_total, P_net, P_IT, em_real, LMS_new, LMC_new, LMI_new

    def run(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {
                'lms': [random.random() for _ in range(T)],
                'lmc': [random.randint(0, T-t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
            }
            pop.append(ind)
            
        best_ind = None; min_fit = float('inf')
        for it in range(self.max_iter):
            fits = []
            for ind in pop:
                f, _, _, _, _, _, _, _, _, _ = self.calculate_fitness(ind)
                fits.append(f)
                if f < min_fit:
                    min_fit = f
                    best_ind = copy.deepcopy(ind)
            
            sorted_idx = np.argsort(fits)
            new_pop = [pop[i] for i in sorted_idx[:self.pop_size//2]]
            while len(new_pop) < self.pop_size:
                c = copy.deepcopy(random.choice(new_pop[:10]))
                if random.random() < 0.7: c['lmc'] = [random.randint(0, T-t['dur']) for t in self.tasks_LMC]
                if random.random() < 0.7: c['lms'][random.randint(0,T-1)] = random.random()
                if random.random() < 0.7: c['lmi'] = [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
                new_pop.append(c)
            pop = new_pop
            
        return best_ind

if __name__ == "__main__":
    LNM, LMS_ori, Price, PV_ori, Wind_ori = load_data()
    

   
    total_it_power = np.sum(LNM) + np.sum(LMS_ori) + (1008.11*8 + 504.05*16) + (403.24*20 + 336.04*16)
    total_dc_power = total_it_power * PUE_CONSTANT
    base_res_power = np.sum(PV_ori) + np.sum(Wind_ori)
    base_ratio = base_res_power / total_dc_power
    

    scale_factor = 0.50 / base_ratio if base_ratio > 0 else 0
    PV = PV_ori * scale_factor
    Wind = Wind_ori * scale_factor
    


    LMC_base = np.zeros(T); LMI_base = np.zeros(T)
    LMC_base[72:72+8] += 1008.11; LMC_base[72:72+16] += 504.05
    LMI_base[72:72+20] += 403.24; LMI_base[72:72+16] += 336.04
    
    P_IT_base = LNM + LMS_ori + LMC_base + LMI_base
    P_total_base = P_IT_base * PUE_CONSTANT
    P_net_base = np.maximum(0, P_total_base - PV - Wind)
    
    Total_RES = PV + Wind
    Curtailed_RES_ori = np.maximum(0, Total_RES - P_total_base)
    curtail_rate_ori = (np.sum(Curtailed_RES_ori) / np.sum(Total_RES)) * 100 if np.sum(Total_RES) > 0 else 0
    
    cost_elec_base = 0; em_real_base = 0
    for t in range(T):
        if P_net_base[t] > 0:
            cost_elec_base += P_net_base[t] * Price[t] * 0.25
            em_real_base += P_net_base[t] * E_GRID_DYNAMIC[t] * 0.25 
            
    net_em_base = max(0, em_real_base - C_QUOTA)
    cost_green_base = 0; cost_carbon_base = 0
    if net_em_base > 0:
        gec_unit_cost = PRICE_GEC / E_GRID_AVG 
        if gec_unit_cost < PRICE_CARBON: cost_green_base = net_em_base * gec_unit_cost
        else:
            if net_em_base <= CARBON_L: cost_carbon_base = net_em_base * PRICE_CARBON
            elif net_em_base <= 2 * CARBON_L: cost_carbon_base = CARBON_L * PRICE_CARBON + (net_em_base - CARBON_L) * PRICE_CARBON * (1 + CARBON_MU)
            else: cost_carbon_base = CARBON_L * PRICE_CARBON * (2 + CARBON_MU) + (net_em_base - 2 * CARBON_L) * PRICE_CARBON * (1 + 2 * CARBON_MU)
                
    cost_op_base = np.sum(PV) * COST_PV * 0.25 + np.sum(Wind) * COST_WIND * 0.25
    F1_ori = cost_elec_base + cost_op_base + cost_green_base + cost_carbon_base
    F2_ori = em_real_base

  
    scheduler_F1 = SchedulerMOO(LNM, LMS_ori, Price, PV, Wind, mode='min_F1')
    best_ind_F1 = scheduler_F1.run()
    _, F1_min, F2_max, _, _, _, _, _, _, _ = scheduler_F1.calculate_fitness(best_ind_F1)
    print(f"  F1_min = {F1_min:.2f},F2_max = {F2_max:.2f}")

    scheduler_F2 = SchedulerMOO(LNM, LMS_ori, Price, PV, Wind, mode='min_F2')
    best_ind_F2 = scheduler_F2.run()
    _, F1_max, F2_min, _, _, _, _, _, _, _ = scheduler_F2.calculate_fitness(best_ind_F2)
    print(f"   F1_max = {F1_max:.2f}, F2_min = {F2_min:.2f}")

 
    scheduler_Final = SchedulerMOO(LNM, LMS_ori, Price, PV, Wind, mode='weighted')
    scheduler_Final.extremes = {'F1_min': F1_min, 'F1_max': F1_max, 'F2_min': F2_min, 'F2_max': F2_max}
    scheduler_Final.w1 = 0.5; scheduler_Final.w2 = 0.5  

    best_ind_Final = scheduler_Final.run()
    _, final_F1, final_F2, P_total_opt, P_net_opt, P_IT_opt, _, LMS_opt, LMC_opt, LMI_opt = scheduler_Final.calculate_fitness(best_ind_Final)
    
    Curtailed_RES_opt = np.maximum(0, Total_RES - P_total_opt)
    curtail_rate_opt = (np.sum(Curtailed_RES_opt) / np.sum(Total_RES)) * 100 if np.sum(Total_RES) > 0 else 0
    
   

    print("="*45)
    print(f"【 F1】 Original: {F1_ori:.2f} CNY  | Optimized: {final_F1:.2f} CNY  | Reduction: {(F1_ori - final_F1)/F1_ori*100:.2f} %")
    print(f"【F2】 Original: {F2_ori:.2f} kg | Optimized: {final_F2:.2f} kg | Reduction: {(F2_ori - final_F2)/F2_ori*100:.2f} %")
    print(f"【Physical Metrics Baseline Curtailment Rate: {curtail_rate_ori:.2f} % | Optimized: {curtail_rate_opt:.2f} %")
    print("="*45 + "\n")
    
  
    LEGEND_FONT = 11
    x = np.arange(0, 24, 0.25)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, LNM, width=0.25, label='LNM', color='#e6194b')
    plt.bar(x, LMS_ori, width=0.25, bottom=LNM, label='LMS', color='#3cb44b')
    plt.bar(x, LMC_base, width=0.25, bottom=LNM+LMS_ori, label='LMC', color='#ffe119')
    plt.bar(x, LMI_base, width=0.25, bottom=LNM+LMS_ori+LMC_base, label='LMI', color='#4363d8')
    plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('IT Power (kW)', fontweight='bold')
    plt.xlim(0, 24); plt.legend(loc='upper left', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    plt.grid(False); plt.savefig('Fig1_Load_Before.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, LNM, width=0.25, label='LNM', color='#e6194b')
    plt.bar(x, LMS_opt, width=0.25, bottom=LNM, label='LMS', color='#3cb44b')
    plt.bar(x, LMC_opt, width=0.25, bottom=LNM+LMS_opt, label='LMC', color='#ffe119')
    plt.bar(x, LMI_opt, width=0.25, bottom=LNM+LMS_opt+LMC_opt, label='LMI', color='#4363d8')
    plt.plot(x, PV+Wind, color='black', linestyle='--', linewidth=2, label='Total RES Generation')
    plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('IT Power (kW)', fontweight='bold')
    plt.xlim(0, 24); plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    plt.grid(False); plt.savefig('Fig2_Load_After.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, PV, color='#ffcc00', linewidth=2, label='PV')
    plt.plot(x, Wind, color='#33cc33', linewidth=2, label='Wind')
    plt.fill_between(x, 0, PV+Wind, color='#99ff99', alpha=0.3, label='Total RES')
    plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('Power (kW)', fontweight='bold')
    plt.xlim(0, 24); plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    plt.grid(False); plt.savefig('Fig3_RES_Profile.png', dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    plt.step(x, Price, where='post', color='#800080', linewidth=2.5, label='TOU Price')
    plt.fill_between(x, 0, Price, step='post', color='#e6ccff', alpha=0.3)
    plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('Price (CNY/kWh)', fontweight='bold')
    plt.xlim(0, 24); plt.ylim(0, 1.4)
    plt.text(3, 0.73, 'Flat (0.68)', color='purple', fontweight='bold', fontsize=11, ha='center')
    plt.text(13, 0.40, 'Valley (0.35)', color='purple', fontweight='bold', fontsize=11, ha='center')
    plt.text(20.5, 1.20, 'Peak (1.15)', color='purple', fontweight='bold', fontsize=11, ha='center')
    plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    plt.grid(False); plt.savefig('Fig4_Price_Curve.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    LMI_diff = LMI_opt - LMI_base 
    

    ax1.bar(x, np.maximum(0, LMI_diff), width=0.25, align='edge', color='#2ca02c', alpha=0.8, label='VES Charging (Load Increase)')
    ax1.bar(x, np.minimum(0, LMI_diff), width=0.25, align='edge', color='#d62728', alpha=0.8, label='VES Discharging (Load Reduction)')
    ax1.step(x, LMI_base, where='post', color='gray', linestyle=':', linewidth=2, label='Baseline LMI (18:00 Peak)')
    ax1.step(x, LMI_opt, where='post', color='#4363d8', linestyle='-', linewidth=2, label='Optimized LMI')
    
    ax1.set_xlabel('Time (h)', fontweight='bold')
    ax1.set_ylabel('Power (kW)', fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.grid(False)

    

    ax2 = ax1.twinx()
    
  
    vSoC_aggregate = np.cumsum(LMI_opt) / np.sum(LMI_opt) * 100
  
    vSoC_plot = np.zeros(len(x))
    vSoC_plot[1:] = vSoC_aggregate[:-1]
    
    ax2.step(x, vSoC_plot, where='post', color='#1f77b4', linewidth=2.5, linestyle='-', label='Aggregate vSoC Tracking (%)')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Virtual State of Charge - vSoC (%)', color='#1f77b4', fontweight='bold')
    
   
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    
    plt.savefig('Fig5_VES_Action.png', dpi=300, bbox_inches='tight')

    print("All Done")