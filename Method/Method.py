import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# Global plotting style settings
# ==========================================
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# Global system parameters
# ==========================================
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
    E_GRID_DYNAMIC[t] = 0.85 if (hour < 8.0 or hour >= 23.0) else 0.35  

# ================================
# 1. Genetic Algorithm with Statistics
# ================================
class SchedulerMOO:
    def __init__(self, LNM, LMS_ori, Price, PV, Wind, tasks_LMC, tasks_LMI):
        self.LNM = LNM
        self.LMS_ori = LMS_ori
        self.Price = Price
        self.PV = PV
        self.Wind = Wind
        self.tasks_LMC = tasks_LMC
        self.tasks_LMI = tasks_LMI
        
        self.pop_size = 100   
        self.max_iter = 300  
        
        self.extremes = {'F1_min': 20000, 'F1_max': 35000, 'F2_min': 10000, 'F2_max': 18000} 
        self.w1 = 0.5; self.w2 = 0.5  

    def calculate_fitness(self, ind):
        LMS_new = (np.array(ind['lms']) / (np.sum(ind['lms']) + 1e-6)) * np.sum(self.LMS_ori)
        LMC_new = np.zeros(T); LMI_new = np.zeros(T); cost_dr = 0.0  
        
        for i, t in enumerate(self.tasks_LMC):
            s = int(ind['lmc'][i]); LMC_new[s:s+t['dur']] += t['p']
            
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
            
        norm_F1 = (F1 - self.extremes['F1_min']) / (self.extremes['F1_max'] - self.extremes['F1_min'] + 1e-6)
        norm_F2 = (F2 - self.extremes['F2_min']) / (self.extremes['F2_max'] - self.extremes['F2_min'] + 1e-6)
        fitness = self.w1 * norm_F1 + self.w2 * norm_F2 + penalty
            
        return fitness, F1, F2

    def run_with_history(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {
                'lms': [random.random() for _ in range(T)],
                'lmc': [random.randint(0, T-t['dur']) for t in self.tasks_LMC],
                'lmi': [[random.random() for _ in range(T)] for _ in self.tasks_LMI]
            }
            pop.append(ind)
            
        best_ind = None; min_fit = float('inf')
        
        history_fitness = []; history_F1 = []; history_F2 = []

        for it in range(self.max_iter):
            fits = []
            for ind in pop:
                f, f1, f2 = self.calculate_fitness(ind)
                fits.append(f)
                if f < min_fit:
                    min_fit = f
                    best_ind = copy.deepcopy(ind)
            
            _, current_best_f1, current_best_f2 = self.calculate_fitness(best_ind)
            history_fitness.append(min_fit)
            history_F1.append(current_best_f1)
            history_F2.append(current_best_f2)
                    
            sorted_idx = np.argsort(fits)
            new_pop = [pop[i] for i in sorted_idx[:self.pop_size//2]]
            
            while len(new_pop) < self.pop_size:
                c = copy.deepcopy(random.choice(new_pop[:max(1, self.pop_size // 5)]))
                
                if random.random() < 0.6: 
                    idx = random.randint(0, len(c['lmc'])-1)
                    shift = random.choice([-4, -2, -1, 1, 2, 4])
                    c['lmc'][idx] = max(0, min(T - self.tasks_LMC[idx]['dur'], c['lmc'][idx] + shift))
                elif random.random() < 0.1:
                    c['lmc'] = [random.randint(0, T-t['dur']) for t in self.tasks_LMC]

                if random.random() < 0.6:
                    c['lms'] = [max(0, v + random.gauss(0, 0.1)) for v in c['lms']]
                
                if random.random() < 0.6:
                    task_idx = random.randint(0, len(c['lmi'])-1)
                    c['lmi'][task_idx] = [max(0, v + random.gauss(0, 0.1)) for v in c['lmi'][task_idx]]

                new_pop.append(c)
            pop = new_pop
            
        return best_ind, history_fitness, history_F1, history_F2

# ================================
# 2. 10 Independent Runs and Statistics Plot
# ================================
if __name__ == "__main__":
    Price = pd.read_csv('TOU.csv')['Price'].values
    
    try:
        df_wl = pd.read_csv('Base_workload.csv')
        df_res = pd.read_csv('res_data.csv')
    except Exception as e:
        print(f"Failed to read CSV files: {e}")
        exit()
    
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
    PV = df_res['PV'].values * scale_factor
    Wind = df_res['Wind'].values * scale_factor

    RUN_TIMES = 10 
    MAX_ITER = 300 
    
    all_history_F1 = np.zeros((RUN_TIMES, MAX_ITER))
    all_history_F2 = np.zeros((RUN_TIMES, MAX_ITER))
    final_F1_list = []
    final_F2_list = []

    print(f"\nStarting algorithm robustness test ({RUN_TIMES} runs, {MAX_ITER} iterations each)...")
    print("Please wait, this may take a while...")
    
    for r in range(RUN_TIMES):
        print(f"  -> Running test {r+1}/{RUN_TIMES}...")
        random.seed(SEED + r)
        np.random.seed(SEED + r)
        
        scheduler = SchedulerMOO(LNM_sc, LMS_sc, Price, PV, Wind, tasks_LMC, tasks_LMI)
        scheduler.max_iter = MAX_ITER
        scheduler.pop_size = 100
        best_ind, h_fit, h_F1, h_F2 = scheduler.run_with_history()
        
        all_history_F1[r, :] = h_F1
        all_history_F2[r, :] = h_F2
        final_F1_list.append(h_F1[-1])
        final_F2_list.append(h_F2[-1])

    # =====================================================
    # 3. Plot Statistical Figures
    # =====================================================
    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(1, 2, 1)
    mean_F1 = np.mean(all_history_F1, axis=0)
    std_F1 = np.std(all_history_F1, axis=0)
    
    iters = np.arange(1, MAX_ITER + 1)
    ax1.plot(iters, mean_F1, color='#1f77b4', linewidth=2.5, label='Mean Total Cost (CNY)')
    ax1.fill_between(iters, mean_F1 - std_F1, mean_F1 + std_F1, color='#1f77b4', alpha=0.2, label='Standard Deviation')
    
    ax1.set_title('(a) Algorithm Convergence Curve', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Generation (Iteration)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Objective Value: Total Cost (CNY)', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='gray')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = plt.subplot(1, 2, 2)
    bp = ax2.boxplot([final_F1_list], positions=[1], widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor='#1f77b4', color='black', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2.5))
                     
    scatter_x = np.random.normal(1, 0.04, size=len(final_F1_list))
    ax2.scatter(scatter_x, final_F1_list, alpha=0.8, color='black', edgecolor='w', s=50, zorder=10)
    
    ax2.set_title('(b) Dispersion of Optimal Solutions', fontweight='bold', fontsize=14)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Optimal F1 (Cost) over 10 runs'], fontweight='bold', fontsize=12)
    ax2.set_ylabel('Total Cost (CNY)', fontweight='bold', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')

    plt.tight_layout()
    img_name = 'FigA4_Algorithm_Statistics.png'
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    print(f"\nStatistical figure generated: {img_name}")
    
    mean_val = np.mean(final_F1_list)
    std_val = np.std(final_F1_list)
    cv_val = (std_val / mean_val) * 100 if mean_val > 0 else 0
    
    print("\n" + "="*50)
    print(" [Statistical Results over 10 Independent Runs]")
    print(f"  F1 (Cost)   Mean: {mean_val:.2f} CNY, Std: {std_val:.2f} CNY")
    print(f"  Coefficient of Variation (CV): {cv_val:.2f} %")
    print(f"  F2 (Carbon) Mean: {np.mean(final_F2_list):.2f} kg")
    print("="*50)