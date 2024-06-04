# %%===========================================================================
# Import
import torch
import pysare
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_preprocess import get_data, normalize_data
from hist_add_noise_merge import add_noise
from seaborn import despine
# %%===========================================================================

# read data
ONE_HOUR = 3600
all_data = pd.read_feather("datapath.feather")
# parameters of reading data and training
config = {
    'feature_names': ['ID', 'voltage', 'ambient_temperature', 'cell_temperature', 'SOC', 'current',
                     'failure_age', 'status'],
    'snapshot_time': 2000,
    'remove_failures': True,
    'production_time': False,
    'failure_snapshot': False,
    'type_change': 'All',
    'censoring': True,
    'batch_size': 512,
    't_m': 1,
    'lr': 1e-3,
    'num_epochs': 10,
    'tail_ratio': 1.2,
    'network_size':[256, 256],
}

# Get the required snapshots

clean_snapshot = get_data(all_data, feature_names=config['feature_names'],
                    snapshot_time=config['snapshot_time'],
                    remove_failures=config['remove_failures'],
                    production_time=config['production_time'],
                    failure_snapshot=config['failure_snapshot'],
                    type_change=config['type_change'],
                    censoring=config['censoring'])

# Adding noise to the bins or merging the histograms 
noisy_features = [item for item in config['feature_names'] if item in ['ambient_temperature', 'cell_temperature', 'SOC', 'current', 'voltage']]
noisy_snapshot = add_noise(clean_snapshot, noisy_features, max_gain=0.0, max_bias = 0, merge_num=4)
# %%===========================================================================
# preprocess the data
snapshot= noisy_snapshot
feature_set = [item for item in snapshot.columns if item not in ['ID', 'status', 'time', 'failure_age', 'index']]
snapshot, normalization_values = normalize_data(snapshot)
snapshot = snapshot.sort_values(by='ID').reset_index(drop=True)
snapshot = snapshot.sample(frac=1, random_state=42)

#split to train validation and test sets and define featureset time and event of the batteries for survival analyses
train_size = int(0.6 * len(snapshot))
validation_size = train_size + int(0.2 * len(snapshot))
train_snapshot = snapshot[:train_size]
validation_snapshot = snapshot[train_size:validation_size]
test_snapshot = snapshot[validation_size:]


train_X, validation_X, test_X = torch.from_numpy(np.array(train_snapshot[feature_set])),\
                                torch.from_numpy(np.array(validation_snapshot[feature_set])),\
                                torch.from_numpy(np.array(test_snapshot[feature_set]))


train_T, validation_T, test_T = train_snapshot['time'].values,\
                                validation_snapshot['time'].values,\
                                test_snapshot['time'].values



train_E, validation_E, test_E = train_snapshot['status'].values,\
                                validation_snapshot['status'].values,\
                                test_snapshot['status'].values


training_set = pysare.data.Dataset(train_X, train_T, train_E)
validation_set = pysare.data.Dataset(validation_X, validation_T, validation_E)
test_set = pysare.data.Dataset(test_X, test_T, test_E)



# Define data loaders
training_loader = torch.utils.data.DataLoader(training_set,
                                              shuffle=True,
                                              batch_size=config['batch_size'])
validation_loader = torch.utils.data.DataLoader(validation_set,
                                                shuffle=False,
                                                batch_size=config['batch_size'])
# %%===========================================================================


# A monte carlo integration scheme with 80 samples is used for estimating gradients
train_integrator = pysare.models.energy_based.integrators.MonteCarlo(80)
# The trapezoidal rule on a uniform grid of 80 points is used for evaluation
eval_integrator = pysare.models.energy_based.integrators.UniformTrapezoidal(80)
# Define model
model = pysare.models.energy_based.EBM.MLP_implementation(
    config['t_m'], config['tail_ratio'], train_integrator, eval_integrator,
    num_features=train_X.size(1), layers=config['network_size'])

# %%===========================================================================
# Train Model

# A torch optimizer is chosen to train the model
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.00)

# Basic training trains the network using negative log-likelihood as loss function
trainnew_model=True
if trainnew_model:
    training_log = pysare.training.basic_training(model, training_loader,
                                                optimizer, num_epochs=config['num_epochs'],
                                                validation_loader=validation_loader,
                                                best_model_checkpoint_path="test.chekpoint")
    # Plot the training
    best_model_state_dict = model.state_dict()
    training_log.plot()
    plt.show()
else:
    checkpoint = torch.load('model.chekpoint')
    model.load_state_dict(checkpoint['model_state_dict'])
# %%===========================================================================
# Evaluate Model
# Evaluation of the model using common evaluation metrics

C_index = pysare.evaluation.concordance_index(model, test_set)
brier = pysare.evaluation.brier_score(model, test_set, num_t=100)
integrated_brier = pysare.evaluation.integrated_brier_score(model, test_set, num_t=100)


# Plot 
plt.subplot(3, 1, 1)
plt.plot(brier['time']+2000, brier['score'])
plt.ylabel('brier')
#plt.title(str(config['type_change']))
plt.subplot(3, 1, 2)
plt.plot(integrated_brier['time']+2000, integrated_brier['integrated_score'] )
plt.ylabel('integrated_brier')

plt.subplot(3, 1, 3)
plt.plot(C_index['time']+2000, C_index['C_index'])

plt.xlabel('Time')
plt.ylabel('C index')
despine()
plt.tight_layout()
plt.savefig('C-index_bier10noise.svg', format='svg')
plt.show()
integrated_brier['integrated_score']
C_index['C_index']


# %%===========================================================================
# Plot Survival functions 

# Here we illustrate the trained model by plotting its survival function andlifetime ensity. 

# Define a vector with times to evaluate the model on
t = np.linspace(0, 1, 100)

# We plot the model for shape parameter
k = 2
l = 2



start_ind = 0
X = test_set.X[start_ind:start_ind+851:20]
T = test_set.T[start_ind:start_ind+851:20]
# Calculate the modeleed survival function and lifetime density
S = model.survival_probability(X, t)
f = model.lifetime_density(X, t)
t = (t * (normalization_values["time"]['max'] - normalization_values["time"]['min'])
     + normalization_values["time"]['min'])/ONE_HOUR+2000



T = (T * (normalization_values["time"]['max'] - normalization_values["time"]['min'])
     + normalization_values["time"]['min'])/ONE_HOUR


fig, ax = plt.subplots()

for n in range(len(T)):
    line, = ax.plot(t, S[:, n])

fig, ax = plt.subplots()

for n in range(len(T)):
    line, = ax.plot(t, f[:, n])


ax.set_xlabel('Time')
ax.set_ylabel('Survival funciton')
ax.legend()
plt.tight_layout()
plt.savefig("survival function 510noise.svg")
plt.show()


