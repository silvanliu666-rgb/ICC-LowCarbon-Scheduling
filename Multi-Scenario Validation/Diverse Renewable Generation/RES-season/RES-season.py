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

try:
    df_season = pd.read_excel('RES-season.xlsx', sheet_name='季节')
except Exception as e:
    print(f"Failed to read file: {e}")
    exit()

scenarios = {
    'Spring': {'cols': ('Spring_PV', 'Spring_Wind', 'Spring_Total_RE'), 'title': '(a) Spring'},
    'Summer': {'cols': ('Summer_PV', 'Summer_Wind', 'Summer_Total_RE'), 'title': '(b) Summer'},
    'Autumn': {'cols': ('Autumn_PV', 'Autumn_Wind', 'Autumn_Total_RE'), 'title': '(c) Autumn'},
    'Winter': {'cols': ('Winter_PV', 'Winter_Wind', 'Winter_Total_RE'), 'title': '(d) Winter'}
}

x = np.arange(0, 24, 0.25)
LEGEND_FONT = 11

for name, config in scenarios.items():
    pv_col, wind_col, total_col = config['cols']
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(x, df_season[pv_col], label='PV', color='#ffcc00', linewidth=2)
    plt.plot(x, df_season[wind_col], label='Wind', color='#33cc33', linewidth=2)
    plt.plot(x, df_season[total_col], label='Total RES', color='black', linestyle='--', linewidth=2)
    
    plt.fill_between(x, 0, df_season[pv_col], color='#ffcc00', alpha=0.15)
    plt.fill_between(x, df_season[pv_col], df_season[total_col], color='#33cc33', alpha=0.15)
    
    plt.title(config['title'], fontweight='bold')
    plt.xlabel('Time (h)', fontweight='bold')
    plt.ylabel('Power (kW)', fontweight='bold')
    plt.xlim(0, 24)
    
    plt.ylim(0, 3500) 
    
    plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9, edgecolor='gray')
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
    
    ax.plot(x, df_season[pv_col], label='PV', color='#ffcc00', linewidth=2)
    ax.plot(x, df_season[wind_col], label='Wind', color='#33cc33', linewidth=2)
    ax.plot(x, df_season[total_col], label='Total RES', color='black', linestyle='--', linewidth=2)
    
    ax.fill_between(x, 0, df_season[pv_col], color='#ffcc00', alpha=0.15)
    ax.fill_between(x, df_season[pv_col], df_season[total_col], color='#33cc33', alpha=0.15)
    
    ax.set_title(config['title'], fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Power (kW)', fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 3500) 
    
    ax.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9, edgecolor='gray')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

combined_file_name = 'Fig8_Seasonal_Generation_Patterns.png'
plt.savefig(combined_file_name, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated combined plot: {combined_file_name}")
print("Seasonal profile plots generated successfully!")