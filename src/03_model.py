
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
data = pd.read_csv('data/processed/raf_features.csv')

# one-hot encode amino acids
def encode_aa(df):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # init arrays for encoding
    wt_encoding = np.zeros((len(df), len(amino_acids)))
    mut_encoding = np.zeros((len(df), len(amino_acids)))

    # create dicts mapping amino acids to positions
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    # fill in encodings
    for i, (wt, mut) in enumerate(zip(df['wt'], df['mut'])):
        if wt in aa_to_index:
            wt_encoding[i, aa_to_index[wt]] = 1
        if mut in aa_to_index:
            mut_encoding[i, aa_to_index[mut]] = 1
    
    # create df cols for one-hot encodings
    wt_cols = [f'wt_{aa}' for aa in amino_acids]
    mut_cols = [f'mut_{aa}' for aa in amino_acids]

    # add encodings to df
    for i, col in enumerate(wt_cols):
        df[col] = wt_encoding[:, i]
    for i, col in enumerate(mut_cols):
        df[col] = mut_encoding[:, i]
    
    return df


# encode amino acids 
# normalize position
