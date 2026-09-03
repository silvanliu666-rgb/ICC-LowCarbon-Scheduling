import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

T = 96               
P_MAX = 5000         
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

E_GRID_DYNAMIC = np.zeros(T)
for t in range(T):
    hour = t * 0.25
    if hour < 8.0 or hour >= 23.0: 
        E_GRID_DYNAMIC[t] = 0.85  
    else: 
        E_GRID_DYNAMIC[t] = 0.35  

class SchedulerMOO:
    def __init__(self, LNM, LMS_ori, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', mode='weighted'):
        self.LNM = LNM
        self.LMS_ori = LMS_ori
        self.Price = Price
        self.PV = PV
        self.Wind = Wind
        self.tasks_LMC = tasks_LMC
        self.tasks_LMI = tasks_LMI
        self.flex_mode = flex_mode 
        
        self.base_starts_lmc = [64, 68, 72, 76, 80, 84, 88, 92]
        self.base_starts_lmi = [56, 64, 72, 80, 68, 72, 76, 80] 
        
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
            gec_unit = PRICE_GEC / E_GRID_AVG 
            if gec_unit < PRICE_CARBON: cost_green = net_em * gec_unit
            else:
                if net_em <= CARBON_L: cost_carbon = net_em * PRICE_CARBON
                elif net_em <= 2*CARBON_L: cost_carbon = CARBON_L*PRICE_CARBON + (net_em-CARBON_L)*PRICE_CARBON*1.25
                else: cost_carbon = CARBON_L*PRICE_CARBON*2.25 + (net_em-2*CARBON_L)*PRICE_CARBON*1.5
                
        cost_op = np.sum(self.PV)*COST_PV*0.25 + np.sum(self.Wind)*COST_WIND*0.25
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
            
        return {'lms': lms_base, 'lmc': lmc_base, 'lmi': lmi_base}

    def enforce_ablation(self, ind):
        base_ind = self.get_baseline_ind()
        
        if self.flex_mode == 'none':
            return base_ind
        elif self.flex_mode == 'only_lms':
            ind['lmc'] = base_ind['lmc']
            ind['lmi'] = base_ind['lmi']
        elif self.flex_mode == 'only_lmi':
            ind['lms'] = base_ind['lms']
            ind['lmc'] = base_ind['lmc']
            
        return ind

    def run(self):
        if self.flex_mode == 'none':
            return self.get_baseline_ind()
            
        pop = []
        for _ in range(self.pop_size):
            ind = {
                'lms': [random.random() for _ in range(T)],
                'lmc': [random.randint(0, T-t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
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
                
                c = self.enforce_ablation(c)
                new_pop.append(c)
                
            pop = new_pop
        return best_ind

if __name__ == "__main__":
    Price = pd.read_csv('TOU.csv')['Price'].values
    
    try:
        df_wl = pd.read_csv('Base_workload.csv')
    except Exception as e:
        print(f"Failed to read Base_workload.csv: {e}")
        exit()
        
    LNM_raw = df_wl['LNM_kW'].values
    LMS_raw = df_wl['LMS_kW'].values
    shape_LNM = LNM_raw / (np.sum(LNM_raw) + 1e-6)
    shape_LMS = LMS_raw / (np.sum(LMS_raw) + 1e-6)
    
    TOTAL_IT_ENERGY = 40324.38486685
    TOTAL_IT_KW_STEPS = TOTAL_IT_ENERGY / 0.25
    
    try:
        df_res = pd.read_csv('res_data.csv')
        PV_ori = df_res['PV'].values
        Wind_ori = df_res['Wind'].values
    except Exception as e:
        print(f"Failed to read res_data.csv: {e}")
        exit()
    
    props = {'LMI': 0.67, 'LMC': 0.11, 'LNM': 0.15, 'LMS': 0.07}
    
    LNM_sc = shape_LNM * (TOTAL_IT_KW_STEPS * props['LNM'])
    LMS_sc = shape_LMS * (TOTAL_IT_KW_STEPS * props['LMS'])
    p_lmc = (TOTAL_IT_KW_STEPS * props['LMC']) / 96.0  
    tasks_LMC = [{'id': i+1, 'p': p_lmc, 'dur': 12} for i in range(8)]
    p_lmi = (TOTAL_IT_KW_STEPS * props['LMI']) / 512.0 
    tasks_LMI = [{'id': i+1, 'p': p_lmi, 'dur': 80} for i in range(4)] + [{'id': i+5, 'p': p_lmi, 'dur': 48} for i in range(4)]
                
    P_total_rough = (np.sum(LNM_sc) + np.sum(LMS_sc) + p_lmc*96 + p_lmi*512) * PUE_CONSTANT
    base_res_power = np.sum(PV_ori) + np.sum(Wind_ori)
    scale_factor = 0.50 / (base_res_power / P_total_rough) if base_res_power > 0 else 0
    PV = PV_ori * scale_factor; Wind = Wind_ori * scale_factor
    Total_RES = PV + Wind

    ablation_scenarios = ['none', 'only_lms', 'only_lmi', 'full']
    names = {
        'none': '1_No_Flexibility',
        'only_lms': '2_Only_LMS_Flexible',
        'only_lmi': '3_Only_LMI_Flexible',
        'full': '4_Full_Workload_Flexibility'
    }
    
    colors_12 = {
        'LNM': '#E64B35', 'LMS': '#F39B7F', 
        'LMC': '#00A087', 'LMI': '#1F77B4', 'RES': '#2F2F4F'
    }
    x = np.arange(0, 24, 0.25)
    results_summary = []
    
    base_F1 = 0; base_F2 = 0

    print("\nStarting Workload Flexibility Ablation Study...")
    print("  -> Calculating Pareto boundaries...")
    
    sch_boundary_1 = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', mode='min_F1')
    _, F1_min, F2_max, _,_,_,_,_,_,_ = sch_boundary_1.calculate_fitness(sch_boundary_1.run())
    sch_boundary_2 = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode='full', mode='min_F2')
    _, F1_max, F2_min, _,_,_,_,_,_,_ = sch_boundary_2.calculate_fitness(sch_boundary_2.run())
    extremes_dict = {'F1_min': F1_min, 'F1_max': F1_max, 'F2_min': F2_min, 'F2_max': F2_max}

    for flex in ablation_scenarios:
        sc_name = names[flex]
        print(f"\nEvaluating: [{sc_name}]...")
        
        scheduler_Final = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI, flex_mode=flex, mode='weighted')
        scheduler_Final.extremes = extremes_dict
        
        best_ind = scheduler_Final.run()
        _, final_F1, final_F2, P_total_opt, _, _, _, LMS_opt, LMC_opt, LMI_opt = scheduler_Final.calculate_fitness(best_ind)
        
        curtail_rate_opt = (np.sum(np.maximum(0, Total_RES - P_total_opt)) / np.sum(Total_RES)) * 100
        
        if flex == 'none':
            base_F1 = final_F1
            base_F2 = final_F2
            
        results_summary.append({
            'Ablation Scenario': sc_name,
            'Op_Cost (CNY)': round(final_F1, 2),
            'Cost_Saved (%)': round((base_F1 - final_F1) / base_F1 * 100, 2) if base_F1 > 0 else 0,
            'Carbon_Em (kg)': round(final_F2, 2),
            'Carbon_Red (%)': round((base_F2 - final_F2) / base_F2 * 100, 2) if base_F2 > 0 else 0,
            'Curtailment (%)': round(curtail_rate_opt, 2)
        })

        plt.figure(figsize=(10, 6))
        plt.bar(x, LNM_sc, width=0.25, align='edge', label='LNM (Rigid)', color=colors_12['LNM'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMS_opt, width=0.25, bottom=LNM_sc, align='edge', label='LMS (Short-term)', color=colors_12['LMS'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMC_opt, width=0.25, bottom=LNM_sc+LMS_opt, align='edge', label='LMC (Continuous)', color=colors_12['LMC'], alpha=0.9, edgecolor='none')
        plt.bar(x, LMI_opt, width=0.25, bottom=LNM_sc+LMS_opt+LMC_opt, align='edge', label='LMI (Interruptible)', color=colors_12['LMI'], alpha=0.9, edgecolor='none')
        plt.plot(x, Total_RES, color=colors_12['RES'], linestyle='--', linewidth=2.5, label='Total RES Generation')
        
        plt.xlabel('Time (h)', fontweight='bold'); plt.ylabel('IT Power (kW)', fontweight='bold')
        plt.xlim(0, 24)
        
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True, fontsize=10, framealpha=0.95, edgecolor='gray')
        plt.grid(False)
        
        if flex == 'only_lms':
            img_name = 'Fig5_Ablation_LMS_only.png'
        elif flex == 'only_lmi':
            img_name = 'Fig6_Ablation_LMI_only.png'
        else:
            img_name = f'Ablation_{flex}.png'
            
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  --> Saved plot: {img_name}")
       
    df_results = pd.DataFrame(results_summary)
    
    print("\n" + "="*90)
    print(" [Workload Flexibility Ablation Study Summary]")
    print("="*90)
    print(df_results.to_markdown(index=False))
    print("="*90 + "\n")
    
    df_results.to_csv('Optimization_Results_Ablation_Summary.csv', index=False)
    print("Ablation study complete. Data saved.")