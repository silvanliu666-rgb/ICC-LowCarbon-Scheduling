import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

T = 96               
P_MAX = 6000         
C_QUOTA = 20000      
PRICE_CARBON = 0.090 
PRICE_GEC = 0.005    
CARBON_MU = 0.25      
CARBON_L = 2000      
PUE_CONSTANT = 1.2    
COST_PV = 0.20        
COST_WIND = 0.15      
COST_DR = 30.0        
E_GRID_AVG = 0.5703 

E_BESS_MAX = 10000.0   
E_BESS_MIN = 1000.0    
P_BESS_MAX = 2500.0    
ETA_C = 0.95           
ETA_D = 0.95           
COST_BESS_DEG = 0.15   

E_GRID_DYNAMIC = np.zeros(T)
for t in range(T):
    hour = t * 0.25
    if hour < 8.0 or hour >= 23.0: 
        E_GRID_DYNAMIC[t] = 0.85  
    else: 
        E_GRID_DYNAMIC[t] = 0.35  

class SchedulerMOO:
    def __init__(self, LNM, LMS_ori, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', bess_mode=False, mode='weighted'):
        self.LNM = LNM
        self.LMS_ori = LMS_ori
        self.Price = Price
        self.PV = PV
        self.Wind = Wind
        self.tasks_LMC = tasks_LMC
        self.tasks_LMI = tasks_LMI
        
        self.flex_mode = flex_mode 
        self.bess_mode = bess_mode 
        self.mode = mode
        
        self.base_starts_lmc = [64, 68, 72, 76, 80, 84, 88, 92]
        self.base_starts_lmi = [56, 64, 72, 80, 68, 72, 76, 80]
        
        self.pop_size = 100   
        self.max_iter = 200  
        self.extremes = {'F1_min': 1.0, 'F1_max': 1.0, 'F2_min': 1.0, 'F2_max': 1.0}
        self.w1 = 0.5; self.w2 = 0.5  

    def get_baseline_ind(self):
        lms_base = self.LMS_ori.copy()
        lmc_base = self.base_starts_lmc.copy()
        lmi_base = []
        for i, t_task in enumerate(self.tasks_LMI):
            p_arr = np.zeros(T)
            start = self.base_starts_lmi[i]
            for j in range(t_task['dur']):
                p_arr[(start + j) % T] = 1.0 
            lmi_base.append(p_arr)
        return {'lms': lms_base, 'lmc': lmc_base, 'lmi': lmi_base, 'bess': [0.0]*T}

    def enforce_ablation(self, ind):
        base_ind = self.get_baseline_ind()
        if self.flex_mode == 'none':
            ind['lms'] = base_ind['lms']
            ind['lmc'] = base_ind['lmc']
            ind['lmi'] = base_ind['lmi']
        if not self.bess_mode:
            ind['bess'] = [0.0] * T
        return ind

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
        P_total_IT = P_IT * PUE_CONSTANT
        
        soc = E_BESS_MAX * 0.2  
        P_bess_charge = np.zeros(T)
        P_bess_discharge = np.zeros(T)
        cost_bess_deg = 0.0
        
        for t in range(T):
            action = ind['bess'][t] 
            if action > 0 and self.bess_mode:
                p_c = action * P_BESS_MAX
                max_c = (E_BESS_MAX - soc) / (0.25 * ETA_C)
                p_c = min(p_c, max_c)
                P_bess_charge[t] = p_c
                soc += p_c * 0.25 * ETA_C
                cost_bess_deg += p_c * 0.25 * COST_BESS_DEG
            elif action < 0 and self.bess_mode:
                p_d = -action * P_BESS_MAX
                max_d = (soc - E_BESS_MIN) * ETA_D / 0.25
                p_d = min(p_d, max_d)
                P_bess_discharge[t] = p_d
                soc -= (p_d * 0.25) / ETA_D
                cost_bess_deg += p_d * 0.25 * COST_BESS_DEG
                
        P_net = np.maximum(0, P_total_IT + P_bess_charge - P_bess_discharge - self.PV - self.Wind)
        
        cost_grid = 0; em_real = 0
        for t in range(T):
            if P_net[t] > 0:
                cost_grid += P_net[t] * self.Price[t] * 0.25
                em_real += P_net[t] * E_GRID_DYNAMIC[t] * 0.25 
        
        net_em = max(0, em_real - C_QUOTA)
        cost_green = 0; cost_carbon = 0
        if net_em > 0:
            gec_unit = PRICE_GEC / E_GRID_AVG 
            if gec_unit < PRICE_CARBON: cost_green = net_em * gec_unit
            else:
                if net_em <= CARBON_L: cost_carbon = net_em * PRICE_CARBON
                elif net_em <= 2*CARBON_L: cost_carbon = CARBON_L*PRICE_CARBON + (net_em-CARBON_L)*PRICE_CARBON*1.25
                else: cost_carbon = CARBON_L*PRICE_CARBON*2.25 + (net_em-2*CARBON_L)*PRICE_CARBON*1.5
                
        cost_op = np.sum(self.PV)*COST_PV*0.25 + np.sum(self.Wind)*COST_WIND*0.25
        
        F1 = cost_grid + cost_op + cost_green + cost_carbon + cost_dr + cost_bess_deg
        F2 = em_real 
        
        max_power_draw = np.max(P_total_IT + P_bess_charge)
        penalty = 1e7 if max_power_draw > P_MAX else 0
            
        if self.mode == 'min_F1': fitness = F1 + penalty
        elif self.mode == 'min_F2': fitness = F2 + penalty
        elif self.mode == 'weighted':
            norm_F1 = (F1 - self.extremes['F1_min']) / (self.extremes['F1_max'] - self.extremes['F1_min'] + 1e-6)
            norm_F2 = (F2 - self.extremes['F2_min']) / (self.extremes['F2_max'] - self.extremes['F2_min'] + 1e-6)
            fitness = self.w1 * norm_F1 + self.w2 * norm_F2 + penalty
            
        return fitness, F1, F2, P_total_IT, P_bess_charge, P_bess_discharge, P_net, LMS_new, LMC_new, LMI_new

    def run(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {
                'lms': [random.random() for _ in range(T)],
                'lmc': [random.randint(0, T-t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI],
                'bess': [random.uniform(-1, 1) for _ in range(T)]
            }
            ind = self.enforce_ablation(ind)
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
                
                if random.random() < 0.8:
                    c['bess'] = [random.uniform(-1, 1) if random.random() < 0.1 else c['bess'][i] for i in range(T)]
                
                c = self.enforce_ablation(c)
                new_pop.append(c)
                
            pop = new_pop
        return best_ind

if __name__ == "__main__":
    try:
        Price = pd.read_csv('TOU.csv')['Price'].values
        df_wl = pd.read_csv('Base_workload.csv')
        df_res = pd.read_csv('res_data.csv')
    except Exception as e:
        print(f"Failed to read files: {e}")
        sys.exit()
    
    LNM_raw = df_wl['LNM_kW'].values
    LMS_raw = df_wl['LMS_kW'].values
    shape_LNM = LNM_raw / (np.sum(LNM_raw) + 1e-6)
    shape_LMS = LMS_raw / (np.sum(LMS_raw) + 1e-6)
    TOTAL_IT_ENERGY = 40324.38486685
    TOTAL_IT_KW_STEPS = TOTAL_IT_ENERGY / 0.25
    
    props = {'LMI': 0.67, 'LMC': 0.11, 'LNM': 0.15, 'LMS': 0.07}
    
    LNM_sc = shape_LNM * (TOTAL_IT_KW_STEPS * props['LNM'])
    LMS_sc = shape_LMS * (TOTAL_IT_KW_STEPS * props['LMS'])
    p_lmc = (TOTAL_IT_KW_STEPS * props['LMC']) / 96.0  
    tasks_LMC = [{'id': i+1, 'p': p_lmc, 'dur': 12} for i in range(8)]
    p_lmi = (TOTAL_IT_KW_STEPS * props['LMI']) / 512.0 
    tasks_LMI = [{'id': i+1, 'p': p_lmi, 'dur': 80} for i in range(4)] + [{'id': i+5, 'p': p_lmi, 'dur': 48} for i in range(4)]
                
    P_total_rough = (np.sum(LNM_sc) + np.sum(LMS_sc) + p_lmc*96 + p_lmi*512) * PUE_CONSTANT
    base_res_power = np.sum(df_res['PV'].values) + np.sum(df_res['Wind'].values)
    scale_factor = 0.50 / (base_res_power / P_total_rough) if base_res_power > 0 else 0
    PV = df_res['PV'].values * scale_factor; Wind = df_res['Wind'].values * scale_factor
    Total_RES = PV + Wind

    scenarios = [
        {'name': '1_No_Flex_No_BESS', 'flex': 'none', 'bess': False},
        {'name': '2_Physical_BESS_Only', 'flex': 'none', 'bess': True},
        {'name': '3_Virtual_Storage_Only', 'flex': 'full', 'bess': False},
        {'name': '4_Combined_BESS_and_Virtual', 'flex': 'full', 'bess': True}
    ]
    
    colors_12 = {'LNM': '#E64B35', 'LMS': '#F39B7F', 'LMC': '#00A087', 'LMI': '#1F77B4', 'RES': '#2F2F4F'}
    x = np.arange(0, 24, 0.25)
    results_summary = []
    
    print("\nCalculating Pareto Extremes...")
    sch_b1 = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', bess_mode=True, mode='min_F1')
    _, F1_min, F2_max, _,_,_,_,_,_,_ = sch_b1.calculate_fitness(sch_b1.run())
    sch_b2 = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', bess_mode=True, mode='min_F2')
    _, F1_max, F2_min, _,_,_,_,_,_,_ = sch_b2.calculate_fitness(sch_b2.run())
    extremes_dict = {'F1_min': F1_min, 'F1_max': F1_max, 'F2_min': F2_min, 'F2_max': F2_max}

    print("\nStarting ablation study (Physical BESS vs Virtual Storage)...")

    for sc in scenarios:
        sc_name = sc['name']
        print(f"\nEvaluating: [{sc_name}]...")
        
        scheduler = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode=sc['flex'], bess_mode=sc['bess'], mode='weighted')
        scheduler.extremes = extremes_dict
        
        best_ind = scheduler.run()
        _, final_F1, final_F2, P_IT, P_bc, P_bd, P_net, LMS_opt, LMC_opt, LMI_opt = scheduler.calculate_fitness(best_ind)
        
        P_total_demand = P_IT + P_bc  
        curtail_rate = (np.sum(np.maximum(0, Total_RES - P_total_demand)) / np.sum(Total_RES)) * 100
        
        results_summary.append({
            'Scenario': sc_name,
            'Total Cost (CNY)': round(final_F1, 2),
            'Total Carbon (kg)': round(final_F2, 2),
            'Curtailment (%)': round(curtail_rate, 2)
        })

        plt.figure(figsize=(10, 6))
        
        plt.bar(x, LNM_sc, width=0.25, align='edge', label='LNM (Rigid)', color=colors_12['LNM'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMS_opt, width=0.25, bottom=LNM_sc, align='edge', label='LMS (Short-term)', color=colors_12['LMS'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMC_opt, width=0.25, bottom=LNM_sc+LMS_opt, align='edge', label='LMC (Continuous)', color=colors_12['LMC'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMI_opt, width=0.25, bottom=LNM_sc+LMS_opt+LMC_opt, align='edge', label='LMI (Virtual Storage)', color=colors_12['LMI'], alpha=0.9, edgecolor='none')
        
        if sc['bess']:
            base_for_bc = LNM_sc + LMS_opt + LMC_opt + LMI_opt
            plt.bar(x, P_bc, width=0.25, bottom=base_for_bc, align='edge', label='Physical BESS Charge (+)', color='#8c564b', alpha=0.8, edgecolor='none')
            plt.bar(x, -P_bd, width=0.25, align='edge', label='Physical BESS Discharge (-)', color='#e377c2', alpha=0.8, edgecolor='none')
            plt.axhline(0, color='black', linewidth=1)
            
        plt.plot(x, Total_RES, color=colors_12['RES'], linestyle='--', linewidth=2.5, label='Total RES Generation')
        
        plt.title(f'Benchmarking: {sc_name}', fontweight='bold')
        plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('Power (kW)', fontweight='bold')
        plt.xlim(0, 24)
        
        if sc['bess']: plt.ylim(-P_BESS_MAX - 500, P_MAX + 1000)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=10)
        plt.grid(False)
        plt.tight_layout()
        
        img_name = f'Profile_{sc_name}.png'
        plt.savefig(img_name, dpi=300)
        plt.close()
        print(f"  --> Saved plot: {img_name}")

    df_results = pd.DataFrame(results_summary)
    print("\n" + "="*85)
    print(" [BESS vs Virtual Storage Comparison Results]")
    print("="*85)
    print(df_results.to_markdown(index=False))
    print("="*85 + "\n")
    df_results.to_csv('Optimization_Results_BESS_Comparison.csv', index=False)
    print("Analysis complete.")