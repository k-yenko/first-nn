# logs

## understanding dataset distributions, experimental setup and scoring system
Weng, C., Faure, A.J., Escobedo, A. et al. The energetic and allosteric landscape for KRAS inhibition. Nature 626, 643–652 (2024). https://doi.org/10.1038/s41586-023-06954-0

what they found: most allosteric mutations weaken KRAS bidning to partners; some selectively weaken only certain partners -> may be possible to fine-tune which downstream pathway is dialed down

KRAS is a relatively "flat" protein, and doesn't offer deep crevices that small molecules like to nestle into, which is why their "partners" (other large proteins) have advantages in binding

* depth/shape: small molecules need deep, well-defined pockets. KRAS mostly has a broad/fairly flat/slightly curved landscape, which provides a relatively large surface area for another protein to lay (like 2 puzzle pieces)
* interaction type: small molecules typically like tight hydrophobic contacts in a tiny space. KRAS provides a distributed network of many weaker contacts, and partner proteins have complementary bumps, grooves, charges, H-bond donors, and acceptors
* energy pay-off: small molecules need a handful of strong interactions. KRAS prefers dozens of moderate interactions, and affinity to KRAS comes from the sum of those small contacts (+ water exclusion)


## checking distribution of data
for binding fitness data, all partners seem to present a bimodal distribution, with block 1 producing more disruptive mutations across different binding partners
* consider additional features that might explain bimodality (protein structure info, physicochemical properties, etc.)
* block as categorical feature?
* block 1 weights different in model architecture?