import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

df = pd.read_excel('RES-domi.xlsx', sheet_name='RES-domi')

scenarios = {
    'S1': {'cols': ('PV', 'Wind', 'total'), 'title': '(a) Normal baseline'},
    'S2': {'cols': ('PV.1', 'Wind.1', 'total.1'), 'title': '(b) PV-dominated'},
    'S3': {'cols': ('PV.2', 'Wind.2', 'total.2'), 'title': '(c) Wind-dominated'},
    'S4': {'cols': ('PV.3', 'Wind.3', 'total.3'), 'title': '(d) High volatility'}
}

x = np.arange(0, 24, 0.25)
LEGEND_FONT = 11

for name, config in scenarios.items():
    pv_col, wind_col, total_col = config['cols']
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, df[pv_col], label='PV', color='#ffcc00', linewidth=2)
    plt.plot(x, df[wind_col], label='Wind', color='#33cc33', linewidth=2)
    plt.plot(x, df[total_col], label='Total RES', color='black', linestyle='--', linewidth=2)
    
    plt.fill_between(x, 0, df[pv_col], color='#ffcc00', alpha=0.15)
    plt.fill_between(x, df[pv_col], df[total_col], color='#33cc33', alpha=0.15)
    
    plt.title(config['title'], fontweight='bold')
    plt.xlabel('Time (h)', fontweight='bold')
    plt.ylabel('Power (kW)', fontweight='bold')
    plt.xlim(0, 24)
    
    plt.ylim(0, 3500) 
    
    plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    file_name = f'Profile_RES_{name}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated individual plot: {file_name}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, config) in enumerate(scenarios.items()):
    ax = axes[i]
    pv_col, wind_col, total_col = config['cols']
    
    ax.plot(x, df[pv_col], label='PV', color='#ffcc00', linewidth=2)
    ax.plot(x, df[wind_col], label='Wind', color='#33cc33', linewidth=2)
    ax.plot(x, df[total_col], label='Total RES', color='black', linestyle='--', linewidth=2)
    
    ax.fill_between(x, 0, df[pv_col], color='#ffcc00', alpha=0.15)
    ax.fill_between(x, df[pv_col], df[total_col], color='#33cc33', alpha=0.15)
    
    ax.set_title(config['title'], fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Power (kW)', fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3500)  
    
    ax.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

combined_file_name = 'Fig9_Extreme_Generation_Patterns.png'
plt.savefig(combined_file_name, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated combined plot: {combined_file_name}")
print("All plots generated successfully!")