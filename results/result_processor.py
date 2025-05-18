import numpy as np
import pandas as pd
import os

current_directory = os.getcwd()
print(current_directory)
subdir = "results"
subdir_path = os.path.join(current_directory, subdir)
# List only directories inside subdir_path
file_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)
              if os.path.isdir(os.path.join(subdir_path, filename))]

dic = ["/metrics", "/pred", "/true"]

# Process each subdirectory path
for path in file_paths:
    result_list = []
    for item in dic:
        data_path = path + item + ".npy"
        if ".py" not in data_path:  # Ignore Python scripts if any
            data = np.load(data_path)
            if "metric" in data_path:
                column_names = ["mae", "mse", "rmse", "mape", "mspe"]
                # Create DataFrame for metrics
                df_metric = pd.DataFrame({column_names[i]: [data[i]] for i in range(len(data))})
                df_metric.to_csv(path + "/metric.csv", sep="\t")
            if "pred" in data_path or "true" in data_path:
                # Note: There might be sliding window issues here, but alignment is enough for now
                # The final output order: IMF0 to error, last feature is the original signal
                num_features = data.shape[-1]
                if num_features == 1:  # Single target case
                    column_names = ["target"]
                elif "imf" in data_path:  # For decomposition experiments
                    column_names = [f'imf_{i}' for i in range(num_features - 1)]
                    column_names.append("error")
                else:  # For all-in-one experiments
                    column_names = [f'imf_{i}' for i in range(num_features - 2)]
                    column_names.append("error")
                    column_names.append("original")
                # Convert 3D array to DataFrame
                df = pd.DataFrame(data.reshape(-1, num_features), columns=column_names)
                # Sum first columns and compare with last column
                if "imf" in data_path:
                    cols_to_sum = df.columns[:-2]
                else:
                    cols_to_sum = df.columns[:-1]
                # For IMF experiments, sum decomposition components
                if "our_imf" in data_path or "kla_imf" in data_path or "cer_imf" in data_path:
                    df["sum_imf"] = df.sum(axis=1)
                if "pred" in data_path:
                    df_pred = df
                    df_pred.to_csv(path + "/pred.csv", sep="\t")
                    result_list.append(df_pred.iloc[:, -1])
                if "true" in data_path:
                    df_true = df
                    df_true.to_csv(path + "/true.csv", sep="\t")
                    result_list.append(df_true.iloc[:, -1])

print("Transformation successful")
