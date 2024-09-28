import PyQt6.QtCore
import os
os.environ["QT_API"] = "pyqt5"
import matplotlib.pyplot as plt

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import psd_array_multitaper
import os
from config import Config
configObj = Config()
configss = configObj.getConfigSnapshot()

# script1_participant_level_pca_scatter.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mne
from mne.time_frequency import psd_array_multitaper
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)

# Assuming EEG data is in MNE format (Epochs or Raw). Placeholder for file path to participants' data
data_dir = "path_to_data"

# Frequency bands definitions
freq_bands = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}

# Placeholder for results
results = []


import os
import mne

def load_participant_epochs(pnum, condition):
    """Load and concatenate epochs for a participant across multiple blocks for a given condition.
    
    Parameters:
    - pnum: Participant number (integer)
    - condition: Condition (e.g., 'D' or 'ND')
    
    Returns:
    - concatenated_epochs: Concatenated MNE epochs object for the participant
    """
    
    # Initialize an empty list to store the epochs from each block
    epochs_list = []
    
    # Loop through all 4 blocks
    for b_cnt in range(0, 4):
        # Create the block numbers by appending the current block count to the condition
        block_num = f"{condition}{b_cnt}"
        
        # Construct the participant's folder name
        participant_number = 'P' + str(pnum)
        # Create the full path to the epochs file
        participant_data_path = participant_number + '/' + block_num + '-epo.fif'
        path = os.path.join(configss['root'], 'reading_task',configss['data_dir'], participant_data_path)

        print(path)
        
        # Load the epochs file for this block
        epochs = mne.read_epochs(path, preload=True)
        
        # Append the loaded epochs to the list
        epochs_list.append(epochs)
    
    # Concatenate all the loaded epochs from the list
    concatenated_epochs = mne.concatenate_epochs(epochs_list)
    
    return concatenated_epochs



# Function to perform time-frequency analysis using multitaper with per-participant p-value corrections
def time_frequency_analysis_multitaper_per_participant(epochs_condition1, epochs_condition2, freq_bands, participant_id):
    """
    [Your existing function code]
    """
    # ... (Use your existing function implementation here)
    # For brevity, it's not repeated here.
    pass  # Replace with your function's code

# Frequency bands to analyze
freq_bands = {
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30)
}

# Initialize lists to collect data for PCA
X = []  # Feature vectors
y = []  # Labels ('D' or 'ND')
participant_ids = []  # Participant IDs
conditions = []  # Condition labels

valid_pids = [el for el in list(range(1, 32)) if el not in [5, 13, 14, 16, 17, 20, 31]]

# Iterate over participants and conditions to collect features
for participant_id in (1,):
    for condition in ['D', 'ND']:
        try:
            epochs = load_participant_epochs(participant_id, condition)
        except NotImplementedError:
            print(f"Data loading function not implemented. Skipping participant {participant_id}, condition {condition}.")
            continue
        
        # Perform time-frequency analysis (if needed)
        # results_df = time_frequency_analysis_multitaper_per_participant(
        #     epochs_condition1=epochs_condition1,
        #     epochs_condition2=epochs_condition2,
        #     freq_bands=freq_bands,
        #     participant_id=participant_id
        # )
        # You can integrate your analysis results here if required
        
        # Example feature extraction: PSD features
        sfreq = epochs.info['sfreq']
        psds, freqs = psd_array_multitaper(
            epochs.get_data(), sfreq=sfreq, fmin=1, fmax=40,
            bandwidth=2.0, adaptive=True, normalization='full', verbose=False
        )
        # psds shape: (n_epochs, n_channels, n_freqs)
        
        # Flatten PSDs for PCA
        psds_mean = psds.mean(axis=-1)  # Average over frequency
        psds_flat = psds_mean.reshape(psds_mean.shape[0], -1)  # Flatten channels and frequencies
        X.append(psds_flat)
        y.extend([condition] * psds_flat.shape[0])
        participant_ids.extend([participant_id] * psds_flat.shape[0])
        conditions.extend([condition] * psds_flat.shape[0])

# Concatenate all feature vectors
X = np.vstack(X)
y = np.array(y)
participant_ids = np.array(participant_ids)
conditions = np.array(conditions)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for plotting
df_pca = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
df_pca['Condition'] = y
df_pca['Participant'] = participant_ids

# Plotting
plt.figure(figsize=(10, 8))
conditions_unique = df_pca['Condition'].unique()
colors = {'D': 'blue', 'ND': 'red'}

for condition in conditions_unique:
    indices = df_pca['Condition'] == condition
    plt.scatter(
        df_pca.loc[indices, 'PCA1'],
        df_pca.loc[indices, 'PCA2'],
        c=colors.get(condition, 'grey'),
        label=condition,
        alpha=0.6,
        edgecolors='w',
        s=50
    )

plt.title('PCA Scatter Plot at Participant Level (Each Epoch)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Condition')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_scatter_participant_level.png', dpi=300)
plt.show()
