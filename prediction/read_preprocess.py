import pandas as pd
import numpy as np

def normalize_data(df):
    normalization_values = {}

    for column in df.columns:
        if column in ['time', 'snapshot_age', 'failure_age', 'age_equivalent_full_cycles', 'maximum_capacity',
                      'age_equivalent_full'] or \
                any(column.startswith(item) for item in ['ambient_temperature', 'cell_temperature', 'SOC', 'current',
                                                         'voltage']):
            # Normalize between 0 and 1
            df[column] = df[column]
            min_value = df[column].min()
            max_value = df[column].max()
            if max_value - min_value != 0:
                normalized_values = (df[column] - min_value) / (max_value - min_value)
            else:
                normalized_values = (df[column] - min_value) / 1
            df[column] = normalized_values
            normalization_values[column] = {'min': min_value, 'max': max_value}

    return df, pd.DataFrame(normalization_values)


def get_data(all_data, feature_names=[], snapshot_time=20000,  remove_failures=False, production_time=False, failure_snapshot=False,
             type_change='All', censoring=False):

    ONE_HOUR = 3600
    bin_num = 100

    if type_change is False:
        all_data = all_data.loc[all_data['TypeChangeTimes'] == 1]
        all_data = all_data.drop(['type_2', 'type_3'], axis=1)
    elif type_change is True:
        all_data = all_data.loc[all_data['TypeChangeTimes'] != 1]

    if 'Type' in feature_names or not feature_names:
        all_data[['type_1', 'type_2', 'type_3']] = pd.get_dummies(all_data.Type, prefix='type').astype(float)
        #all_data = all_data.drop("type", axis=1)
    # generates feature set that should be returned
    if not feature_names:
        feature_set = all_data.columns
    else:
        feature_set = []
        for feature in feature_names:
            if feature in ['ambient_temperature', 'cell_temperature', 'SOC', 'current', 'voltage']:
                feature_set.extend([feature + '_' + str(i+1) for i in range(bin_num)])
            elif feature == "Type":
                feature_set.extend(['type_1', 'type_2', 'type_3'])
            else:
                feature_set.append(feature)
    # returns only the failure or the last snapshot
    if failure_snapshot is True:
        all_data = all_data.loc[all_data['status'] == 1]
    # includes the production_time also
    if production_time is True:
        all_data["time_now"] = np.where(all_data['snapshot_age'] - all_data['production'] > 0,
                                        all_data['snapshot_age'] - all_data['production'], 0)
        data = all_data.loc[all_data['time_now'] / ONE_HOUR == snapshot_time]
        data = data.loc[:, feature_set]

    else:
        if remove_failures is True:
            data = all_data.loc[all_data['snapshot_age'] / ONE_HOUR == snapshot_time]
        else:
            data = all_data.loc[all_data['snapshot_age'] / ONE_HOUR <= snapshot_time]
            data = data.loc[data.groupby('ID')['snapshot_age'].idxmax()]
        data = data.loc[:, feature_set]
        data = data.reset_index()

    if censoring is True:
        l = len(data)
        np.random.seed(42)
        data['censoring_time'] = np.random.randint(snapshot_time * ONE_HOUR, 14000 * ONE_HOUR, l)
        data['status'] = np.where(data['failure_age'] >= data['censoring_time'], 0, 1)
        data['time'] = np.where(data['status'] == 1, (data['failure_age']) - snapshot_time*ONE_HOUR,
                                data['censoring_time']-snapshot_time*ONE_HOUR)
        drop_col = [item for item in ['censoring_time', 'failure_age'] if item in data.columns]
        data = data.drop(drop_col, axis=1)

    else:
        data = data.rename(columns={'failure_age': 'time'})
    return data
