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

## choosing a "partner"
i used RAF1RDB as a starting point. this was kind of an arbitrary choice, but also because disruptions in the KRAS-RAF1 are relatively well-documented so in case i decide to probe this path further, there's some helpful literature. 

## addressing a "bug"
or my just my lack of look at the full dataset. during model training, at every 20th epoch, loss and MSE would be NaN. was my learning rate set too high, causing model outputs to become inf/nan and thus loss of function? were there outliers in the data, somehow causing my model to break? 

i realized the solution was simple (and an oversight on my part): i forgot the data included double mutants (i.e. not just `p.Val125Leu` but also p.`Glu63Lys;Leu113Phe`). the fix was to modify the regex function to handle double mutants. i decided to keep the double mutants, even if that meant refactoring what was already written, since most of the data was actually double mutation data, and it seemed like the most biologically meaningful approach. the following changes were made:

1. rewrote the parser to extract the wt, pos, and mut details from a second mutation, if applicable
2. fixed-size encoding: reserved two "slots" for mutations, where each slot was encoded with:
    a. one-hot for WT AA
    b. one-hot for mutant AA
    c. normalized position
    and if there was only one mutation, the second slots was filled with a bunch of zeros (aka "no mutation")
3. added mutation count feature: single v. double mutants have different impacts on protein function, so adding this as a feature allowed for the model to account for this difference. 

## fin! 
it ran! results:
```
Feature stats: 0.0 2.0 0.07568363315081336 0.2956770909132856
Target stats: -1.45034831532026 0.285723457825962 -0.45698406900606603 0.39863930934731184
epoch 20/200, loss: 0.0720, MAE: 0.2089, val loss: 0.0602, val MAE: 0.1870
epoch 40/200, loss: 0.0574, MAE: 0.1826, val loss: 0.0492, val MAE: 0.1663
epoch 60/200, loss: 0.0515, MAE: 0.1717, val loss: 0.0461, val MAE: 0.1585
epoch 80/200, loss: 0.0488, MAE: 0.1673, val loss: 0.0444, val MAE: 0.1547
epoch 100/200, loss: 0.0465, MAE: 0.1628, val loss: 0.0425, val MAE: 0.1507
epoch 120/200, loss: 0.0447, MAE: 0.1594, val loss: 0.0419, val MAE: 0.1486
epoch 140/200, loss: 0.0435, MAE: 0.1576, val loss: 0.0398, val MAE: 0.1438
epoch 160/200, loss: 0.0418, MAE: 0.1539, val loss: 0.0393, val MAE: 0.1414
epoch 180/200, loss: 0.0404, MAE: 0.1509, val loss: 0.0380, val MAE: 0.1399
epoch 200/200, loss: 0.0399, MAE: 0.1506, val loss: 0.0372, val MAE: 0.1376
test MAE: 0.1376
Feature stats: 0.0 2.0 0.07568363315081336 0.2956770909132856
Target stats: -1.45034831532026 0.285723457825962 -0.45698406900606603 0.39863930934731184
```

## some rough conclusions and interpretations
* test MAE of 0.1376 means that, on average, my model's predictions are off by about 0.14 units of binding fitness score
* both the training and validation loss/MAE decrease steadily, which is a good sign for training that's stable
* validation and test MAE are very close, which means the model's performance on the test set is similar to its performance on the validation set, indicating that it's learning patterns and not just memorizing the validation set (or overfitting)
* relatively low MAE suggests the model can capture meaningful patterns in how mutations (single and double) affect KRAS-RAF1RDB binding

## further improvements
1. model complexity: i did one of the more simple sequential NN architectures (linear layers - reLU activations - dropout), but i could probably improve performance by experimenting with more complex architectures
2. feature importance: look into which features (positions, AAs, mutation count) are most influential (maybe using SHAP to quantify the contribution of each feature to the prediction)
3. error analysis: which mutants have the largest errors (i.e. positional, double mutants, etc.)?
4. biological validation: compare known disruptive KRAS mutations vs. those that are just benign with the predictions of the model
5. model extensions: add additional features (i.e. structural/biophysical properties)