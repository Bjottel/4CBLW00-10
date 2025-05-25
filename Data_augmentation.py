import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import random
from typing import List
import pandas as pd
import json
from collections import defaultdict
import matplotlib.pyplot as plt



################################################################################
                                  # VARIATIONS
################################################################################


def horizontal_shift(x: np.ndarray, y: np.ndarray, max_shift: int = 20) -> np.ndarray:
    """
    Shift spectrum left/right along the x-axis by interpolating the y-values.
    """
    delta = random.randint(-max_shift, max_shift)
    x_shifted = x + delta
    f = interp1d(x_shifted, y, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    return f(x)


def vertical_noise(y: np.ndarray, scale: float = 0.05) -> np.ndarray:
    """
    Add small vertical noise to the spectrum to simulate measurement fluctuations.
    """
    noise = (1 - y) * np.random.uniform(-scale, scale, size=y.shape)
    
    return np.clip(y + noise, 0, 1)


def linear_comb(y_list: List, weights: float = None) -> np.ndarray:
    """
    Combine multiple spectra from the same functional group.
    """
    y_array = np.array(y_list)
    
    if weights is None:
        weights = np.random.dirichlet(np.ones(len(y_list)), size=1)[0]
    
    return np.dot(weights, y_array)


def intensity_scaling(y: np.ndarray, gamma_range=(0.9, 1.1)) -> np.ndarray:
    """
    Randomly scale the entire spectrum's intensity.
    """
    gamma = np.random.uniform(*gamma_range)
    
    return gamma * y


def sinusoidal_drift(x: np.ndarray, y: np.ndarray, amp_range=(0.005, 0.02), freq_range=(0.5, 2)) -> np.ndarray:
    """
    Add a sinusoidal drift to the spectrum to simulate baseline distortions.
    """
    A = np.random.uniform(*amp_range)
    f = np.random.uniform(*freq_range) / (x[-1] - x[0])  # normalize over x range
    phi = np.random.uniform(0, 2 * np.pi)
    drift = A * np.sin(2 * np.pi * f * x + phi)
    
    return y + drift


def smoothing_variation(y: np.ndarray, window_choices=[5, 7, 9, 11], polyorder=2) -> np.ndarray:
    """
    Smooth the spectrum using a Savitzky-Golay filter with a random window size.
    """
    window_size = np.random.choice(window_choices)
    
    if len(y) >= window_size:
        return savgol_filter(y, window_length=window_size, polyorder=polyorder)
    
    return y


################################################################################
                             # AUGMENTING DATASET
################################################################################


# Apply random combinations of augmentations
def augment_randomly(df, x, n_aug=5):
    augmented_spectra = []
    augmented_labels = []
    origin_ids = [] # keep track of which original spectra we perform the augmentations on

    # Group spectra by label for linear combination
    label_to_spectra = defaultdict(list)
    for idx, row in df.iterrows():
        label_str = json.dumps(row['labels'])
        label_to_spectra[label_str].append(np.array(row['spectrum']))

    # Randomly augment each sample
    for idx, row in df.iterrows():
        original_y = np.array(row['spectrum'])
        label = row['labels']
        label_str = json.dumps(label)
        origin_id = f"sample_{idx}"

        augmented_spectra.append(json.dumps(original_y.tolist()))
        augmented_labels.append(json.dumps(label))
        origin_ids.append(origin_id)

        for _ in range(n_aug):
            y_aug = original_y.copy()
            
            # Apply random augmentations
            if random.random() < 0.5:
                y_aug = horizontal_shift(x, y_aug)
            if random.random() < 0.5:
                y_aug = vertical_noise(y_aug)
            if random.random() < 0.5:
                y_aug = intensity_scaling(y_aug)
            if random.random() < 0.5:
                y_aug = sinusoidal_drift(x, y_aug)
            if random.random() < 0.5:
                y_aug = smoothing_variation(y_aug)

            # 50% chance of doing a linear combination instead of above
            if random.random() < 0.5 and len(label_to_spectra[label_str]) >= 2:
                y_pool = label_to_spectra[label_str]
                sampled = random.sample(y_pool, k=min(3, len(y_pool)))
                y_aug = linear_comb(sampled)

            # Clip final result to valid range
            y_aug = np.clip(y_aug, 0, 1)

            augmented_spectra.append(json.dumps(y_aug.tolist()))
            augmented_labels.append(json.dumps(label))
            origin_ids.append(origin_id)

    return pd.DataFrame({
        'spectrum': augmented_spectra,
        'labels': augmented_labels,
        'origin_id': origin_ids
    })


# Apply individual augmentations
def augment_individually(df, x):
    AUG_METHODS = ['horizontal_shift', 'vertical_noise', 'intensity_scaling',
                   'sinusoidal_drift', 'smoothing_variation', 'linear_comb']

    augmented_spectra = []
    augmented_labels = []
    origin_ids = [] # keep track of which original spectra we perform the augmentations on
    augmentation_types = []

    # Group spectra by label for linear combination
    label_to_spectra = defaultdict(list)
    for idx, row in df.iterrows():
        label_str = json.dumps(row['labels'])
        label_to_spectra[label_str].append(np.array(row['spectrum']))

    # Apply all augmentations individually to each sample
    for idx, row in df.iterrows():
        original_y = np.array(row['spectrum'])
        label = row['labels']
        label_str = json.dumps(label)
        origin_id = f"sample_{idx}"

        augmented_spectra.append(json.dumps(original_y.tolist()))
        augmented_labels.append(json.dumps(label))
        origin_ids.append(origin_id)
        augmentation_types.append('none')

        for method in AUG_METHODS:
            y_aug = original_y.copy()

            if method == 'horizontal_shift':
                y_aug = horizontal_shift(x, y_aug)
            elif method == 'vertical_noise':
                y_aug = vertical_noise(y_aug)
            elif method == 'intensity_scaling':
                y_aug = intensity_scaling(y_aug)
            elif method == 'sinusoidal_drift':
                y_aug = sinusoidal_drift(x, y_aug)
            elif method == 'smoothing_variation':
                y_aug = smoothing_variation(y_aug)
            elif method == 'linear_comb':
                y_pool = label_to_spectra[label_str]
                if len(y_pool) >= 2:
                    sampled = random.sample(y_pool, k=min(3, len(y_pool)))
                    y_aug = linear_comb(sampled)
                else:
                    continue

            # Clip final result to valid range            
            y_aug = np.clip(y_aug, 0, 1)

            augmented_spectra.append(json.dumps(y_aug.tolist()))
            augmented_labels.append(json.dumps(label))
            origin_ids.append(origin_id)
            augmentation_types.append(method)

    return pd.DataFrame({
        'spectrum': augmented_spectra,
        'labels': augmented_labels,
        'origin_id': origin_ids,
        'augmentation_type': augmentation_types
    })


ORIGINAL_CSV = 'spectra.csv'
AUGMENTED_CSV = 'augmented_spectra.csv'
IND_AUGMENTED_CSV = 'individual_augmented_spectra.csv'

df = pd.read_csv(ORIGINAL_CSV)
df['spectrum'] = df['spectrum'].apply(json.loads)
df['labels'] = df['labels'].apply(json.loads)

# Determine x based on one spectrum
MIN_WAVENUMBER = 400
MAX_WAVENUMBER = 4000
NUM_POINTS = len(df.iloc[0]['spectrum'])
x = np.linspace(4000, 400, NUM_POINTS)

aug_df = augment_randomly(df, x, n_aug=5)
aug_df.to_csv(AUGMENTED_CSV, index=False)

ind_aug_df = augment_individually(df, x)
ind_aug_df.to_csv(IND_AUGMENTED_CSV, index=False)


################################################################################
                         # VISUALIZING AUGMENTATIONS
################################################################################


def plot_combined_augmentations(csv_file, id_number):
    df_aug = pd.read_csv(csv_file)
    df_aug['spectrum'] = df_aug['spectrum'].apply(json.loads)
    
    example_id = df_aug['origin_id'].iloc[id_number]
    example_rows = df_aug[df_aug['origin_id'] == example_id].reset_index(drop=True)
    
    original_y = np.array(example_rows.iloc[0]['spectrum'])
    augmented_y_list = [np.array(row['spectrum']) for _, row in example_rows.iloc[1:].iterrows()]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, original_y, label='Original', linewidth=2)
    for i, y_aug in enumerate(augmented_y_list):
        plt.plot(x, y_aug, label=f'Augmented #{i+1}', alpha=0.7)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorbance')
    plt.title(f'Origin ID: {example_id} - Original vs Combined Augmentations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()


def plot_individual_augmentations(csv_file, id_number):
    df_aug = pd.read_csv(csv_file)
    df_aug['spectrum'] = df_aug['spectrum'].apply(json.loads)
    
    example_id = df_aug['origin_id'].iloc[id_number]
    example_rows = df_aug[df_aug['origin_id'] == example_id].reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    for _, row in example_rows.iterrows():
        spectrum = np.array(row['spectrum'])
        method = row['augmentation_type']
        lw = 2 if method == 'none' else 1
        plt.plot(x, spectrum, label=method, linewidth=lw, alpha=0.8)
    
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorbance')
    plt.title(f'Origin ID: {example_id} - Original vs Individual Augmentations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()


plot_combined_augmentations(AUGMENTED_CSV, id_number=3)
plot_individual_augmentations(IND_AUGMENTED_CSV, id_number=1)
