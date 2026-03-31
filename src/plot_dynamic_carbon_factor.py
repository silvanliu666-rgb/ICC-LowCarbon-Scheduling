import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

x = np.arange(0, 24, 0.25)
E_GRID_DYNAMIC = np.zeros(96)
for t in range(96):
    hour = t * 0.25
    if hour < 8.0 or hour >= 23.0: 
        E_GRID_DYNAMIC[t] = 0.85  
    else: 
        E_GRID_DYNAMIC[t] = 0.35  

LEGEND_FONT = 15 

plt.figure(figsize=(10, 6))

line_color = '#d62728'
fill_color = '#ff9896'

plt.step(x, E_GRID_DYNAMIC, where='post', color=line_color, linewidth=2.5, label='Dynamic Emission Factor')
plt.fill_between(x, 0, E_GRID_DYNAMIC, step='post', color=fill_color, alpha=0.3)

plt.xlabel('Time (h)', fontweight='bold')
plt.ylabel('Grid Emission Factor in kgCO2 per kWh', fontweight='bold')
plt.xlim(0, 24)
plt.ylim(0, 1.0)

plt.text(4, 0.88, 'High Carbon 0.85', color=line_color, fontweight='bold', fontsize=11, ha='center')
plt.text(15.5, 0.38, 'Low Carbon 0.35', color=line_color, fontweight='bold', fontsize=11, ha='center')

plt.legend(loc='upper right', frameon=True, fontsize=LEGEND_FONT, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig('Fig6_Carbon_Curve.png', dpi=300, bbox_inches='tight')