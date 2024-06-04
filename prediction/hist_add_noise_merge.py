# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %%
def CSV2feather():
    all_data = pd.read_csv("CSV file path.csv", header=0)
    for i in range(1,9):  
        temp = pd.read_csv("other CSV files path" + str(i) + ".csv", header=0)
        all_data = pd.concat([all_data, temp], axis=0)
    all_data.to_feather("feather file path.feather")

def compute_new_histogram(old_edges, old_counts_df, new_edges):
    old_counts = old_counts_df.values
    old_start = np.array(old_edges[:-1]).reshape(-1, 1)
    old_end = np.array(old_edges[1:]).reshape(-1, 1)
    new_start = np.array(new_edges[:-1])
    new_end = np.array(new_edges[1:])
    
    overlap_start = np.maximum(old_start, new_start)
    overlap_end = np.minimum(old_end, new_end)
    
    overlap_width = np.maximum(0, overlap_end - overlap_start)
    bin_widths = (old_end - old_start)
    
    # Compute the ratio matrix
    ratio_matrix = overlap_width / bin_widths
    
    # Compute the new histogram counts
    new_counts = np.dot(old_counts, ratio_matrix)
    
    return new_counts

def merge_bins( old_bin_counts, num_bins_to_merge):
    
    # Sum the counts using np.add.reduceat
    new_bin_counts = np.add.reduceat(old_bin_counts, np.arange(0, len(old_bin_counts), num_bins_to_merge))
    

    
    return new_bin_counts

# %%
def add_noise(alldata, features, max_gain, max_bias, merge_num, constant_noise = False):
    original_bin_MinMaxNum = {
        'ambient_temperature'   : [-25, 45, 100],
        'cell_temperature'      : [-25, 65, 100],
        'SOC'                   : [0, 100, 100],
        'current'               : [-40, 40, 100],
        'voltage'               : [12, 15.5, 100],
        }

    noisy_bin_MinMaxNum = {k: [ v[0]* (1 + max_gain) - max_bias,
                                v[1]* (1 + max_gain) + max_bias, 
                                np.ceil(v[2] * (1 + max_gain) + max_bias * 2)]for k, v in original_bin_MinMaxNum.items()}
    np.random.seed(42)
    noisy_df= alldata[[col for col in alldata.columns if not any(col.startswith(feature) for feature in features)]]
    feature_column_map = {feature: [col for col in alldata.columns if col.startswith(feature)] for feature in features}



    for feature in features: 
        print(feature)
        new_edges, edges , new_edges_merged = generate_edges(feature, original_bin_MinMaxNum, max_gain, max_bias, merge_num)

        all_counts_one_feature = None
        column_names = [feature + '_' +str(i) for i in range(1, len(new_edges_merged))]
        noisy_df[column_names] = None
        for id in alldata['ID'].unique():
            one_vehicle_hist = alldata[alldata['ID'] == id][feature_column_map[feature]]
            
            gain = np.random.uniform(-max_gain, max_gain, 1)
            bias = np.random.uniform(-max_bias, max_bias, 1)
            if constant_noise:
                gain =  max_gain
                bias = max_bias

            noisy_edges = edges * (1 + gain) + bias

            new_counts_series = one_vehicle_hist.apply(lambda row: compute_new_histogram(noisy_edges, row, new_edges), axis=1)
            
            if merge_num != 1:

                new_counts_merged  = new_counts_series.apply(lambda row: merge_bins( row, merge_num))
            else:
                new_counts_merged = new_counts_series

            if all_counts_one_feature is None:
                all_counts_one_feature = np.stack(new_counts_merged.values)
            else:
                all_counts_one_feature = np.concatenate((all_counts_one_feature, np.stack(new_counts_merged.values)))
        
        column_names = [feature + '_' + str(i) for i in range(1, len(new_counts_merged.values[0])+1)]
        noisy_df[column_names] = all_counts_one_feature

    return noisy_df

def generate_edges(feature, original_bin_MinMaxNum, max_gain, max_bias, merge_num):
    noisy_bin_MinMaxNum = {k: [ v[0]* (1 + max_gain) - max_bias,
                            v[1]* (1 + max_gain) + max_bias, 
                            np.ceil(v[2] * (1 + max_gain) + max_bias * 2)]for k, v in original_bin_MinMaxNum.items()}
    new_edges = np.linspace(noisy_bin_MinMaxNum[feature][0], noisy_bin_MinMaxNum[feature][1], int(noisy_bin_MinMaxNum[feature][2]+1))
    edges = np.linspace(original_bin_MinMaxNum[feature][0], original_bin_MinMaxNum[feature][1], int(original_bin_MinMaxNum[feature][2]+1))
    new_edges_merged = new_edges[::merge_num]

    return new_edges, edges, new_edges_merged

