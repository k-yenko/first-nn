import requests
from bs4 import BeautifulSoup

urns = [
    "urn:mavedb:00000115-a-1",
    "urn:mavedb:00000115-a-2",
]

base_url = "https://mavedb.org/score-sets/"

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

