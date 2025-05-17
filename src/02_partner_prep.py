import pickle
import pandas as pd

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



