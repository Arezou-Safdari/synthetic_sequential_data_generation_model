# Synthetic sequential data

In this repository, the code and the data for the paper "Synthetic Generation of Streamed and Snapshot Data for Predictive Maintenance" by Arezou Safdari, Erik Frisk, Olov Holmer, and Mattias Krysander are published.

The paper presents a benchmark synthetic run-to-failure dataset of Li-ion batteries inspired by real-world industrial data. 
The generated data can be used to conduct controlled data-driven prognostics investigations and develop and evaluate data-driven predictive maintenance models.

A detailed description of the dataset generation process is available in the paper, which was accepted and published in the SAFEPROCESS 2024 conference.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Generating data](#generating-data)
- [Using the dataset](#using-the-dataset)



## Dataset

You can download the data in either .csv or .feather formats using this [[link](https://liuonline-my.sharepoint.com/:f:/g/personal/aresa40_liu_se/Em1T3eVLfTlCm6uubcOqxEIBNvSkRah-9k_K0TfztOkOKw?e=MJwlQl)]

### Dataset Structure

Each row in the dataset is a snapshot from lifetime of the vehicle and includes the following columns:

- **ID**: A unique identifier for each individual.
- **failure_age**: The failure time of the individual(ground truth).
- **status**: The event at the snapshot (0 if censored, 1 if failed; it's 1 at the last snapshot).
- **production**: The time when the individual was produced.
- **InitialCapacity**: The initial capacity of the battery (a manufacturing property).
- **climate**: The climate type of the individual.
- **Type**: The usage profile of the component at that snapshot.
- **TypeChangeTimes**: The number of times the vehicle changed type.
- **snapshot_age**: The age of the individual when the snapshot was taken.
- **maximum_capacity**: The maximum capacity of the battery when the snapshot was taken.
- **ambient_temperature_x**: The histogtogram of the ambient temperature.
- **cell_temperature_x**: The histogtogram of the cell temperature.
- **SOC_x**: The histogtogram of the SOC.
- **current_x**:The histogtogram of the current.
- **voltage_x**:The histogtogram of the voltage.

## Generating data

To run the simulation and generate the synthetic time series data, you need to clone the project and run the `data_generator` file.

## Using the dataset

You can find examples of using the dataset for cox propotional hazard and energy based methood in the `prediction` file, however here we provide the implementation of the Kaplan-Meier algorithm. The code is written in Python 3.

### Kaplan meier
The Kaplan-Meier algorithm is implemented as follow. 

First, import the required packages:
import required packages 
```python
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np
```

Download the dataset through this [[link](https://liuonline-my.sharepoint.com/:f:/g/personal/aresa40_liu_se/Em1T3eVLfTlCm6uubcOqxEIBNvSkRah-9k_K0TfztOkOKw?e=MJwlQl)] and read the dataset in python: 

```python
all_data = pd.read_feather("C:/My_files/RAPIDS/Predictions_on_Synthetic_Data/Data/all_data.feather")
```

Select the snapshots that did not have a type change and get the failure time in hours:

```python
ONE_HOUR = 3600
snapshot = all_data.loc[all_data['status'] == 1]
snapshot = snapshot.loc[snapshot['TypeChangeTimes'] == 0]
snapshot.loc[:, "failure_age"] = snapshot["failure_age"] / ONE_HOUR
t = np.linspace(snapshot["failure_age"].min(), snapshot["failure_age"].max(), 1500)
```

Estimate and plot the survival function prediction for each usage profile:

```python
plt.figure()
types = ['type1', 'type2', 'type3']
survival_function = pd.DataFrame()
for i, type in enumerate(types):
    type_i = snapshot[snapshot['Type'] == i+1]
    kmf = KaplanMeierFitter()
    kmf.fit(type_i["failure_age"], type_i["status"])
    kmf.plot_survival_function(title='Kaplan_Meier')
plt.legend(["Type1", "Type1", "Type2", "Type2", "Type3", "Type3"])
plt.show()
```

