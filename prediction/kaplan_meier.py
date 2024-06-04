import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np
# =====================
ONE_HOUR = 3600
# read data
all_data = pd.read_feather("data")

# prepare data
snapshot = all_data.loc[all_data['status'] == 1]
snapshot = snapshot.loc[snapshot['TypeChangeTimes'] == 0]
snapshot.loc[:, "failure_age"] = snapshot["failure_age"] / ONE_HOUR
t = np.linspace(snapshot["failure_age"].min(), snapshot["failure_age"].max(), 1500)
# Kaplan Meier for each type
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


# %%
