import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

with open('data/processed/all_dataframes.pkl', 'rb') as f:
    all_dataframes = pickle.load(f)

# directory for figures
os.makedirs('figures', exist_ok=True)

# individual urn distributions
for urn, df in all_dataframes.items():
    if 'score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['score'], kde=True)
        
        # mean and median lines
        plt.axvline(df['score'].mean(), color='red', linestyle='--', label=f'Mean: {df["score"].mean():.2f}')
        plt.axvline(df['score'].median(), color='green', linestyle=':', label=f'Median: {df["score"].median():.2f}')
        
        # title and labels
        plt.title(f'Distribution of Scores: {urn}')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
        
        # save fig
        plt.savefig(f'figures/{urn.replace(":", "_")}_distribution.png')
        plt.close()