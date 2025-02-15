#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().system('pip install mne numpy pandas matplotlib')
import mne
# Load the EDF file
file_path = "C:/Users/Ramya Sundaram/Downloads/chb-mit-scalp-eeg-database-1.0.0/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf"  # Update with your actual path
raw = mne.io.read_raw_edf(file_path, preload=True)
# Display basic file info
print(raw.info)


# In[26]:


# Define seizure event
onset = 2996  # Seizure start time in seconds
duration = 3036 - 2996  # Duration of seizure
description = ["Seizure"]  # Label

# Create annotation
annotations = mne.Annotations(onset=[onset], duration=[duration], description=description)
raw.set_annotations(annotations)

# Plot the EEG data with the labeled seizure event
raw.plot(duration=60, scalings="auto")
# Reset measurement date to avoid timestamp issues
raw.set_meas_date(None)

# Save the file with a valid MNE-compliant name
raw.save("labeled_chb01_03_raw.fif", overwrite=True)
raw = mne.io.read_raw_fif("labeled_chb01_03_raw.fif", preload=True)
fig=raw.plot()




# In[19]:


import mne

# Load the labeled EEG file
file_path = "C:/Users/Ramya Sundaram/labeled_chb01_03_raw.fif"
raw = mne.io.read_raw_fif(file_path, preload=True)

# Print info to check annotations
print(raw.annotations)


# In[20]:


import pandas as pd
# Find the first seizure annotation
seizure_start = raw.annotations.onset[0]  # Start time in seconds
seizure_duration = raw.annotations.duration[0]  # Duration in seconds

# Convert to sample index
sfreq = raw.info['sfreq']  # Sampling frequency
start_sample = int(seizure_start * sfreq)
end_sample = start_sample + int(seizure_duration * sfreq)

# Extract seizure data
seizure_data, seizure_times = raw[:, start_sample:end_sample]

# Convert to DataFrame
seizure_df = pd.DataFrame(seizure_data, index=raw.ch_names, columns=seizure_times)

# Display the first few rows of the seizure data
print(seizure_df.head())


# In[21]:


import numpy as np
import mne
from mne.time_frequency import psd_array_welch

# Load the raw EEG data
raw = mne.io.read_raw_fif('C:/Users/Ramya Sundaram/labeled_chb01_03_raw.fif', preload=True)

# Extract the data and channel names
data, times = raw[:, :][0], raw.times

# Define frequency bands
freq_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 40)}

# Get power spectral density using Welch's method
psd, freqs = psd_array_welch(data, sfreq=raw.info['sfreq'], fmin=1, fmax=40, n_jobs=1)

# Extract power for each frequency band
band_powers = {}
for band, (fmin, fmax) in freq_bands.items():
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_powers[band] = np.mean(psd[:, idx], axis=1)  # Mean power in the frequency band

# The `band_powers` will contain the power for each frequency band across all channels


# In[22]:


import matplotlib.pyplot as plt
# Apply bandpass filter to isolate relevant frequency bands (e.g., delta, theta, alpha)
raw.filter(l_freq=0.5, h_freq=30)  # for delta band (0.5 - 4 Hz)
# Calculate PSD for the raw EEG data using Welch's method
psd, freqs = psd_array_welch(data, sfreq=raw.info['sfreq'], fmin=0, fmax=50, n_jobs=1)
# Delta (0.5 - 4 Hz), Theta (4 - 8 Hz), and Alpha (8 - 12 Hz)
delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)

# Calculate the power for each frequency band
delta_power = np.mean(psd[:, delta_idx], axis=1)
theta_power = np.mean(psd[:, theta_idx], axis=1)
alpha_power = np.mean(psd[:, alpha_idx], axis=1)
# Compute dynamic thresholds using mean + 2*std deviation
delta_threshold = np.mean(delta_power) + 2 * np.std(delta_power)
theta_threshold = np.mean(theta_power) + 2 * np.std(theta_power)
alpha_threshold = np.mean(alpha_power) - 2 * np.std(alpha_power)  # Alpha should drop
# Detect seizure onset based on thresholds
seizure_indices = (delta_power > delta_threshold) | (theta_power > theta_threshold) | (alpha_power < alpha_threshold)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(delta_power, label='Delta Power (0.5-4 Hz)')
plt.plot(theta_power, label='Theta Power (4-8 Hz)')
plt.plot(alpha_power, label='Alpha Power (8-12 Hz)')
plt.axhline(delta_threshold, color='red', linestyle='--', label='Delta Threshold')
plt.axhline(theta_threshold, color='blue', linestyle='--', label='Theta Threshold')
plt.axhline(alpha_threshold, color='green', linestyle='--', label='Alpha Threshold (Drop)')
plt.scatter(np.where(seizure_indices), delta_power[seizure_indices], color='red', label='Seizure Points')
plt.legend()
plt.title('Power in Frequency Bands & Seizure Detection')
plt.xlabel('Time (Epochs or Seconds)')
plt.ylabel('Power (uV^2/Hz)')
plt.show()
"""Compare the power values (e.g., delta vs alpha power)
print("Delta Power:", delta_power)
print("Theta Power:", theta_power)
print("Alpha Power:", alpha_power)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

# Plot the power of the frequency bands over time
plt.plot(delta_power, label='Delta Power')
plt.plot(theta_power, label='Theta Power')
plt.plot(alpha_power, label='Alpha Power')
plt.legend()
plt.title('Power in Frequency Bands Over Time')
plt.xlabel('Time (in epochs or seconds)')
plt.ylabel('Power (uV^2/Hz)')
plt.show()
threshold_delta = np.mean(delta_power) + 2 * np.std(delta_power)
seizure_onset = np.where(delta_power > threshold_delta)[0]

# Plot the results of seizure detection
plt.plot(delta_power)
plt.axhline(threshold_delta, color='r', linestyle='--', label='Threshold')
plt.scatter(seizure_onset, delta_power[seizure_onset], color='r', label='Seizure Onset')
plt.legend()
plt.show()"""


# In[6]:


import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Utility Functions
# ---------------------------
def binary_transform(x):
    """
    Convert a continuous vector x to a binary vector using a threshold of 0.5.
    """
    return np.where(x > 0.5, 1, 0)

def dummy_classifier_evaluate(X_selected, y, classifier):
    """
    Evaluate classifier performance on the selected feature subset.
    
    Replace this dummy function with your actual classifier evaluation.  
    For demonstration, we perform 5-fold cross-validation using the classifier.
    """
    scores = cross_val_score(classifier, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

def evaluate_fitness(position, X, y, classifier, alpha=0.01):
    """
    Evaluate the fitness of a feature subset.
    
    The fitness is computed as:
         fitness = accuracy - alpha * (number of features selected / total features)
    This encourages high accuracy with fewer features.
    """
    # Select features based on the binary vector 'position'
    selected_indices = np.where(position == 1)[0]
    
    # Return a very poor fitness if no features are selected.
    if len(selected_indices) == 0:
        return 0
    
    X_selected = X[:, selected_indices]
    
    # Evaluate classifier accuracy via cross-validation
    accuracy = dummy_classifier_evaluate(X_selected, y, classifier)
    
    # Penalize the number of features selected
    penalty = alpha * (len(selected_indices) / len(position))
    fitness = accuracy - penalty
    
    return fitness

def exploration_operator(position, mutation_rate=0.1):
    """
    HHO-inspired exploration operator: randomly flip bits in the position vector
    with a small mutation rate.
    """
    new_pos = position.copy()
    for i in range(len(new_pos)):
        if np.random.rand() < mutation_rate:
            new_pos[i] = 1 - new_pos[i]  # Flip the bit
    return new_pos

def exploitation_operator(position, gbest, prob=0.5):
    """
    HHO-inspired exploitation operator: for each bit, if it differs from the global best,
    with a given probability adopt the global best bit.
    """
    new_pos = position.copy()
    for i in range(len(new_pos)):
        if new_pos[i] != gbest[i] and np.random.rand() < prob:
            new_pos[i] = gbest[i]
    return new_pos

# ---------------------------
# Hybrid PSO-HHO Feature Selection Algorithm
# ---------------------------
def hybrid_pso_hho_feature_selection(num_features, X, y, classifier, 
                                       population_size=30, max_iter=50, 
                                       w=0.7, c1=1.5, c2=1.5, hho_prob=0.3):
    """
    Hybrid PSO-HHO Feature Selection algorithm.
    
    Parameters:
      - num_features: Total number of features.
      - X, y: Data and labels.
      - classifier: A scikit-learn classifier instance.
      - population_size: Number of candidate solutions.
      - max_iter: Maximum number of iterations.
      - w: Inertia weight.
      - c1: Cognitive coefficient.
      - c2: Social coefficient.
      - hho_prob: Probability of applying the HHO-inspired operator.
      
    Returns:
      - gbest: Best feature subset found (binary vector).
      - gbest_fitness: Fitness value of gbest.
    """
    # Initialize population (binary vectors) and velocities (continuous values)
    population = [np.random.randint(0, 2, num_features) for _ in range(population_size)]
    velocities = [np.random.rand(num_features) for _ in range(population_size)]
    
    # Initialize personal bests (pbest) and fitness values
    pbest = population.copy()
    pbest_fitness = [evaluate_fitness(ind, X, y, classifier) for ind in population]
    
    # Determine global best (gbest)
    best_index = np.argmax(pbest_fitness)
    gbest = pbest[best_index].copy()
    gbest_fitness = pbest_fitness[best_index]
    
    for it in range(max_iter):
        for i in range(population_size):
            # Standard PSO velocity update:
            r1 = np.random.rand(num_features)
            r2 = np.random.rand(num_features)
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - population[i]) +
                             c2 * r2 * (gbest - population[i]))
            
            # Update position: add velocity to current position and convert to binary
            new_position = binary_transform(population[i] + velocities[i])
            
            # With probability hho_prob, apply HHO-inspired operators:
            if np.random.rand() < hho_prob:
                if np.random.rand() < 0.5:
                    new_position = exploration_operator(new_position)
                else:
                    new_position = exploitation_operator(new_position, gbest)
            
            # Evaluate fitness of the new position
            fitness = evaluate_fitness(new_position, X, y, classifier)
            
            # Update personal best if improvement is found
            if fitness > pbest_fitness[i]:
                pbest[i] = new_position.copy()
                pbest_fitness[i] = fitness
            
            # Update global best if improvement is found
            if fitness > gbest_fitness:
                gbest = new_position.copy()
                gbest_fitness = fitness
            
            # Update particle position
            population[i] = new_position.copy()
        
        print(f"Iteration {it+1}/{max_iter}: Global best fitness = {gbest_fitness:.4f}")
    
    return gbest, gbest_fitness

# ---------------------------
# Example usage
# ---------------------------
if __name__ == '__main__':
    # For demonstration, we generate dummy data.
    num_features = 50  # Total number of features (e.g., EEG features)
    X_dummy = np.random.rand(100, num_features)  # 100 samples with 50 features each
    y_dummy = np.random.randint(0, 2, 100)         # Binary labels (e.g., seizure vs. non-seizure)
    
    # Use a real classifier from scikit-learn, for example RandomForestClassifier.
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Run the hybrid PSO-HHO feature selection algorithm.
    best_features, best_fit = hybrid_pso_hho_feature_selection(num_features, X_dummy, y_dummy, classifier,
                                                               population_size=30, max_iter=50,
                                                               w=0.7, c1=1.5, c2=1.5, hho_prob=0.3)
    
    print("Best feature subset found (binary vector):", best_features)
    print("Best fitness achieved:", best_fit)


# In[16]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def dummy_classifier_evaluate(X_selected, y, classifier):
    scores = cross_val_score(classifier, X_selected, y, cv=5, scoring='accuracy')
    return scores.mean()

def evaluate_fitness(position, X, y, classifier, alpha=0.01):
    selected_indices = np.where(position == 1)[0]
    if len(selected_indices) == 0:
        return 0
    X_selected = X[:, selected_indices]
    accuracy = dummy_classifier_evaluate(X_selected, y, classifier)
    penalty = alpha * (len(selected_indices) / len(position))
    fitness = accuracy - penalty
    return fitness

# Create dummy data for demonstration
np.random.seed(42)
num_samples = 9000
num_features = 50
X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, num_samples)
best_features = np.random.randint(0, 2, num_features)  # Dummy binary vector for feature selection

classifier = RandomForestClassifier(n_estimators=50, random_state=42)

# Evaluate fitness with the dummy best_features
fitness_value = evaluate_fitness(best_features, X, y, classifier)
print("Fitness value:", fitness_value)


# In[9]:


num_selected_features = np.sum(best_features)
print("Number of selected features:", num_selected_features)


# In[11]:


get_ipython().system('pip install tensorflow')
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# For demonstration, create a dummy selected features matrix X_selected
num_samples = X_selected.shape[0]
num_selected_features = X_selected.shape[1]
X_selected = np.random.rand(num_samples, num_selected_features)  # shape (100, 20)
y = np.random.randint(0, 2, num_samples)  # Binary labels (e.g., seizure vs non-seizure)

# Now, define num_selected_features based on X_selected's shape (if not already defined)
#num_selected_features = X_selected.shape[1]

# If you want to treat your data as sequential for an LSTM,
# you need to reshape it into (num_samples, timesteps, feature_dim).
# For example, if you want 10 timesteps:
timesteps = 7

# Ensure that num_selected_features is divisible by timesteps.
# Otherwise, adjust timesteps or pad/truncate the features.
feature_dim = num_selected_features // timesteps  

# Reshape X_selected into a sequential format:
X_seq = X_selected.reshape(num_samples, timesteps, feature_dim)
print("X_seq shape:", X_seq.shape)  # Expected shape

# Build an LSTM model for classification:
model_seq = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, feature_dim)),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # For binary classification
])

model_seq.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_seq.summary()

# Train the model (using a validation split for demonstration):
history_seq = model_seq.fit(X_seq, y, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model:
loss, accuracy = model_seq.evaluate(X_seq, y)
print("LSTM model accuracy on sequential selected features:", accuracy)


# In[14]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install keras-tuner')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Dummy data for demonstration (replace with your actual X_selected and y)
np.random.seed(42)
num_samples = 1000
num_selected_features = 50  # Ensure this is divisible by timesteps
X_selected = np.random.rand(num_samples, num_selected_features)
y = np.random.randint(0, 2, num_samples)

# Normalize the data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_selected)

# Define timesteps and compute feature dimension per timestep
timesteps = 10
feature_dim = num_selected_features // timesteps  # 50 // 10 = 5

# Reshape the data for LSTM: (num_samples, timesteps, feature_dim)
X_seq = X_norm.reshape(num_samples, timesteps, feature_dim)

# Split into training and validation sets
X_train_seq, X_val_seq, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, random_state=42)

def build_lstm_model(hp):
    model = Sequential()
    # First LSTM layer with tunable number of units
    model.add(LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=(timesteps, feature_dim)))
    model.add(Dropout(hp.Float('dropout_lstm', min_value=0.2, max_value=0.5, step=0.1)))
    # Second LSTM layer without return_sequences
    model.add(LSTM(hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)))
    model.add(Dropout(hp.Float('dropout_lstm_2', min_value=0.2, max_value=0.5, step=0.1)))
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='lstm_tuner',
    project_name='EEG_LSTM'
)

lstm_tuner.search(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val), callbacks=[early_stop])
best_lstm_hp = lstm_tuner.get_best_hyperparameters(num_trials=1)[0]
best_lstm_model = lstm_tuner.hypermodel.build(best_lstm_hp)

history_lstm = best_lstm_model.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val), callbacks=[early_stop])
lstm_loss, lstm_accuracy = best_lstm_model.evaluate(X_val_seq, y_val)
print("LSTM model validation accuracy:", lstm_accuracy)


# In[15]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# For demonstration, create dummy data:
num_samples = 1000
num_selected_features = 50  # Example: 50 selected features from feature selection
X_selected = np.random.rand(num_samples, num_selected_features)
y = np.random.randint(0, 2, num_samples)

# Normalize the data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_selected)

# Choose timesteps such that num_selected_features is divisible by timesteps
# Here, let's choose timesteps = 10 (so feature_dim = 50 // 10 = 5)
timesteps = 10
feature_dim = num_selected_features // timesteps  # 50 // 10 = 5

# Reshape into sequential format for LSTM: (samples, timesteps, feature_dim)
X_seq = X_norm.reshape(num_samples, timesteps, feature_dim)

# Split into training and validation sets
X_train_seq, X_val_seq, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, random_state=42)

# Build an improved LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, feature_dim)),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Use EarlyStopping to monitor validation loss
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train_seq, y_train, epochs=100, batch_size=32, validation_data=(X_val_seq, y_val), callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_val_seq, y_val)
print("Improved LSTM model validation accuracy:", accuracy)


# In[27]:


# Load the .fif file
raw = mne.io.read_raw_fif("C:/Users/Ramya Sundaram/labeled_chb01_03_raw.fif", preload=True)

# Convert to DataFrame
df = raw.to_data_frame()

# Save to CSV
df.to_csv("C:/Users/Ramya Sundaram/labeled_chb01_03_raw.csv", index=False)
print(df.columns)

