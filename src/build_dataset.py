import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import os
import re
import random

urns = [
    "urn:mavedb:00000115-a-1",
    "urn:mavedb:00000115-a-2",
]

base_url = "https://mavedb.org/score-sets/"

# data directory
data_dir = 'data/raw'

# organize datasets by type
datasets = {
    "binding_free_energy": {},
    "binding_fitness": {},
    "abundance_fitness": {},
    "folding_free_energy": {}
}

# map urns to categories and binding partners
urn_categories = {}
urn_partners = {}
urn_blocks = {}

# manual experiment detail to map urns
urn_to_description = {
    "urn:mavedb:00000115-a-1": "binding free energy changes of KRAS-DARPin K55 interaction",
    "urn:mavedb:00000115-a-2": "binding free energy changes of KRAS-DARPin K27 interaction",
    "urn:mavedb:00000115-a-3": "binding free energy changes of KRAS-SOS1 interaction",
    "urn:mavedb:00000115-a-4": "binding free energy changes of KRAS-PIK3CGRBD interaction",
    "urn:mavedb:00000115-a-5": "binding free energy changes of KRAS-RALGDSRBD interaction",
    "urn:mavedb:00000115-a-6": "binding free energy changes of KRAS-RAF1RBD interaction",
    "urn:mavedb:00000115-a-7": "folding free energy changes of KRAS in abundancePCA",
    "urn:mavedb:00000115-a-8": "binding fitness from KRAS-DARPin K55 bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-9": "binding fitness from KRAS-DARPin K55 bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-10": "binding fitness from KRAS-DARPin K27 bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-11": "binding fitness from KRAS-DARPin K27 bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-12": "binding fitness from KRAS-DARPin K27 bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-13": "binding fitness from KRAS-SOS1 bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-14": "binding fitness from KRAS-SOS1 bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-15": "binding fitness from KRAS-SOS1 bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-16": "binding fitness from KRAS-PIK3CGRDB bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-17": "binding fitness from KRAS-PIK3CGRDB bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-18": "binding fitness from KRAS-PIK3CGRDB bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-19": "binding fitness from KRAS-RALGDSRDB bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-20": "binding fitness from KRAS-RALGDSRDB bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-21": "binding fitness from KRAS-RALGDSRDB bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-22": "binding fitness from KRAS-RAF1RDB bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-23": "abundance fitness from abundancePCA of KRAS block1",
    "urn:mavedb:00000115-a-24": "binding fitness from KRAS-RAF1RDB bindingPCA of KRAS block3",
    "urn:mavedb:00000115-a-25": "binding fitness from KRAS-RAF1RDB bindingPCA of KRAS block2",
    "urn:mavedb:00000115-a-26": "binding fitness from KRAS-DARPin K55 bindingPCA of KRAS block1",
    "urn:mavedb:00000115-a-27": "abundance fitness from abundancePCA of KRAS block2",  
    "urn:mavedb:00000115-a-28": "abundance fitness from abundancePCA of KRAS block3"
    
}

for urn in urns:
    url = base_url + urn

# populate mapping dicts from description
for urn, description in urn_to_description.items():
    if "binding free energy" in description:
        category = "binding_free_energy"
        partner = description.split("KRAS-")[1].split(" interaction")[0]
        block = None
    elif "folding free energy" in description:
        category = "folding_free_energy"
        partner = None
        block = None
    elif "binding fitness" in description:
        category = "binding_fitness"
        partner = description.split("KRAS-")[1].split(" bindingPCA")[0]
        block_match = re.search(r'block(\d+)', description)
        block = block_match.group(1) if block_match else None
    elif "abundance fitness" in description:
        category = "abundance_fitness"
        partner = None
        block_match = re.search(r'block(\d+)', description)
        block = block_match.group(1) if block_match else None

    # store in mapping dicts
    urn_categories[urn] = category
    urn_partners[urn] = partner
    urn_blocks[urn] = block

# map urns to file paths
def find_file_for_urn(urn):
    """create mapping from filenames to urns"""
    expected_filename = urn.replace(":", "_") + "_scores.csv"
    filepath = os.path.join(data_dir, expected_filename)
    return filepath if os.path.exists(filepath) else None

# load dataframes
all_dataframes = {}

# process each urn
for urn in urn_to_description.keys():
    # find the file
    filepath = find_file_for_urn(urn)
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['urn'] = urn
        df['category'] = urn_categories[urn]
        if urn_partners[urn]:
            df['partner'] = urn_partners[urn]
        if urn_blocks[urn]:
            df['block'] = urn_blocks[urn]
        # store in appropriate dicts
        all_dataframes[urn] = df
        datasets[urn_categories[urn]][urn] = df

# view dataframe structures

print("\n=== loaded dataframe structures ===")
for urn, df in all_dataframes.items():
    print(f"{urn}: {df.shape[0]} rows × {df.shape[1]} columns | cat: {urn_categories[urn]} | partner: {urn_partners.get(urn, 'None')} | block: {urn_blocks.get(urn, 'None')}")
    
# sample df
if all_dataframes:
    # Get a random URN
    sample_urn = random.choice(list(all_dataframes.keys()))
    df = all_dataframes[sample_urn]
    
    print(f"\n=== random sample df ===")
    print(f"urn: {sample_urn}")
    print(f"description: {urn_to_description[sample_urn]}")
    print(f"shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"partner: {urn_partners.get(sample_urn, 'None')}")
    print(df.head(5))
else:
    print("no df loaded")
