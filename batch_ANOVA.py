import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (6, 4)

plt.rcParams.update({
    'font.size': 10,            # Global font size
    'axes.titlesize': 10,       # Title font size
    'axes.labelsize': 8,        # Axis label font size
    'xtick.labelsize': 8,       # X-axis tick label font size
    'ytick.labelsize': 8,       # Y-axis tick label font size
    'legend.fontsize': 10,      # Legend font size
    'figure.titlesize': 10      # Overall figure title font size
})

# Load data (desensitized path)
original_data = pd.read_csv("./data/DYG_new.csv", header=0)

# Extract columns 'jn', 'nd', 'ht'
jn = original_data['jn']
nd = original_data['nd']
ht = original_data['ht']

# Define batch size (100 samples per batch)
batch_size = 100

# Calculate number of batches
num_batches = len(jn) // batch_size

# Split data into batches using numpy's array_split
batches = np.array_split(jn, num_batches)

# Calculate mean and variance for each batch
means = []
variances = []

for batch in batches:
    mean = np.mean(batch)
    variance = np.var(batch)
    means.append(mean)
    variances.append(variance)

# Calculate mean of variances
mean_variance = np.mean(variances)

# Compute difference between each batch variance and mean variance
batch_diff = [(i + 1, abs(variance - mean_variance)) for i, variance in enumerate(variances)]

# Sort batches by difference, select top 10 closest to mean variance
sorted_batches = sorted(batch_diff, key=lambda x: x[1])

# Get the top 10 batches based on variance proximity
top_10_batches = [batches[batch[0] - 1] for batch in sorted_batches[:10]]  # Adjust index (0-based)

# Plot boxplots for the top 10 batches
plt.boxplot(top_10_batches, labels=[f'Batch_{batch[0]}' for batch in sorted_batches[:10]])
plt.title(r"Concentration ($\\mathcal{C}$) Variance Distribution Across Different Batches")
plt.ylabel("Concentration")
plt.xlabel("Different Batches")

# Optionally set y-axis limits
# plt.ylim(0.3, 2)  # Adjust according to your data range

# Save figure (desensitized path)
# plt.savefig(r"./figures/top_10_boxplot.jpg", dpi=600, bbox_inches='tight')

plt.show()
