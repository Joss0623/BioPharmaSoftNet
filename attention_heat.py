import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path (desensitized - replace with your actual path)
file_path = "./data/weights.xlsx"

# Read Excel file sheet named 'ht'
df = pd.read_excel(file_path, sheet_name='ht')

print(df)

# Select subset of data for visualization
spatial_att = df.iloc[0:20, 0:6]

# Configure matplotlib parameters for high-resolution output
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (5, 4)

plt.rcParams.update({
    'font.size': 10,            # Global font size
    'axes.titlesize': 10,       # Title font size
    'axes.labelsize': 10,       # Axis label font size
    'xtick.labelsize': 9,       # X-axis tick label font size
    'ytick.labelsize': 9,       # Y-axis tick label font size
    'legend.fontsize': 9,       # Legend font size
    'figure.titlesize': 10      # Overall figure title font size
})

# Set font family to Times New Roman
plt.rc('font', family='Times New Roman')

# Plot heatmap using seaborn
sns.heatmap(spatial_att, cmap='GnBu', annot=False, cbar=True, vmin=0, vmax=1)
plt.xlabel("Channels")
plt.ylabel("Time steps")
plt.title(r"Attention Tensor of Sugar ($\mathcal{S}$)")

# Save figure (desensitized path)
save_path = "./figures/FIGheatmap_ht.png"
plt.savefig(save_path, bbox_inches='tight')

plt.show()
