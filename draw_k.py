import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot parameters
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (15, 4)  # Figure size
plt.rcParams.update({
    'font.size': 14,            # Global font size
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 14,       # Axis label font size
    'xtick.labelsize': 12,      # X-axis tick label font size
    'ytick.labelsize': 12,      # Y-axis tick label font size
    'legend.fontsize': 12,      # Legend font size
    'figure.titlesize': 16      # Overall figure title font size
})

# File path (desensitized)
file_path = "./data/BestK.xlsx"

# Create figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

def plot_sheet(ax, sheet_name, title, start_row, end_row, start_col, end_col, K_labels, width):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Data from sheet '{sheet_name}':\n", df)

        # Check and adjust index bounds
        if end_row > len(df):
            end_row = len(df)
        if end_col > len(df.columns):
            end_col = len(df.columns)

        # Extract data range
        data_to_plot = df.iloc[start_row:end_row, start_col:end_col]
        print(f"Data to plot from sheet '{sheet_name}':\n", data_to_plot)

        if data_to_plot.empty or data_to_plot.shape[1] < 3:
            print(f"Sheet '{sheet_name}' does not contain enough data to plot.")
            return

        gold_medal = data_to_plot.iloc[:, 0]
        silver_medal = data_to_plot.iloc[:, 1]
        bronze_medal = data_to_plot.iloc[:, 2]

        x = np.arange(len(K_labels))

        bars_gold = ax.bar(x[:len(gold_medal)] - width, gold_medal, width=width, color="#ACD2C7", label="MAE")
        bars_silver = ax.bar(x[:len(silver_medal)], silver_medal, width=width, color="#D57B70", label="MSE")
        bars_bronze = ax.bar(x[:len(bronze_medal)] + width, bronze_medal, width=width, color="#94B5D8", label="RMSE")

        ax.set_xticks(x)
        ax.set_xticklabels(labels=K_labels)
        ax.set_title(title)

        # Annotate bars with values
        for bar in bars_gold:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), va='bottom', ha='center', fontsize=8)
        for bar in bars_silver:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), va='bottom', ha='center', fontsize=8)
        for bar in bars_bronze:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), va='bottom', ha='center', fontsize=8)

    except Exception as e:
        print(f"An error occurred while processing sheet '{sheet_name}': {e}")

# Plot for each sheet
width = 0.2
plot_sheet(axs[0], "jn", r"Concentration ($\mathcal{C}$)", start_row=0, end_row=4, start_col=1, end_col=4,
           K_labels=['K = 3', 'K = 4', 'K = 5', 'K = 6'], width=width)
plot_sheet(axs[1], "nd", r"Viscosity ($\mathcal{V}$)", start_row=0, end_row=3, start_col=1, end_col=4,
           K_labels=['K = 3', 'K = 4', 'K = 5'], width=width)
plot_sheet(axs[2], "ht", r"Sugar ($\mathcal{S}$)", start_row=0, end_row=4, start_col=1, end_col=4,
           K_labels=['K = 3', 'K = 4', 'K = 5', 'K = 6'], width=width)

plt.legend()
plt.tight_layout()

# Save figure (desensitized path)
save_path = "./figures/FIG_K.jpg"
plt.savefig(save_path, bbox_inches='tight')

plt.show()
