import pickle
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

with open('data/processed/all_dataframes.pkl', 'rb') as f:
    all_dataframes = pickle.load(f)

# focusing in on RAF 
raf_urns = [
    "urn:mavedb:00000115-a-22",  # raf block1
    "urn:mavedb:00000115-a-25",  # raf block2
    "urn:mavedb:00000115-a-24"   # raf block3
]

# extract and combine dfs
raf_dfs = [all_dataframes[urn] for urn in raf_urns if urn in all_dataframes]
if raf_dfs:
    combined_raf_df = pd.concat(raf_dfs, ignore_index=True)

    # save
    combined_raf_df.to_csv('data/processed/combined_raf_binding_fitness.csv', index=False)
    print(f"combined raf1 dataset with {len(combined_raf_df)} rows saved to data/processed/combined_raf_binding_fitness.csv")
else:
    print("raf dfs not found")

## prepare features

# parse hgvs strings - extract WT, mutated, and position

def parse_hgvs(hgvs_str):
    """parse HGVS string to extract WT, mutated, and position of amino acids"""
    # map 3 letter amino acid codes to 1 letter
    aa_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    # extract parts using regex
    match = re.match(r'p\.([A-Z][a-z][a-z])(\d+)([A-Z][a-z][a-z])', hgvs_str)
    if match:
        wt_three, position, mut_three = match.groups()
        wt = aa_map.get(wt_three, 'X')
        mut = aa_map.get(mut_three, 'X')
        position = int(position)
        return wt, position, mut
    return None, None, None

# parse and create feature cols
combined_raf_df[['WT', 'position', 'mut']] = combined_raf_df.apply(lambda row: pd.Series(parse_hgvs(row['hgvs_pro'])), axis=1)

# save
combined_raf_df.to_csv('data/processed/raf_features.csv', index=False)

# address bimodal distrubtion in raf binding fitness data
from sklearn.mixture import GaussianMixture

# fit gaussian mixture model
gmm = GaussianMixture(n_components=2)
combined_raf_df['cluster'] = gmm.fit_predict(combined_raf_df[['score']])

# visualize clusters
plt.figure(figsize=(10, 6))
sns.histplot(combined_raf_df, x='score', hue='cluster', kde=True, bins=30)
plt.title('RAF1RDB binding fitness score clusters')
plt.savefig('figures/raf_score_clusters.png')
plt.close()

# save clustered data
combined_raf_df.to_csv('data/processed/raf_clustered.csv', index=False)
print("added cluster labels and saved to data/processed/raf_clustered.csv")