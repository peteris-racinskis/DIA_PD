# DIA_PD: datamining course assignment

The goal of this project is estimating the amount of bias applied as a function of voter country - participant country, aka the "neighbour rule", a supposed phenomenon whereby neighbouring or otherwise friendly countries inflate each other's scores in the contest. Most of the code is ugly, badly optimized and possibly full of duplication. Either way, it's mostly just high-level scripting so code quality was not imperative.

LOOK FOR FULL-SIZED IMAGES IN THE tex/ DIRECTORY

Direct links
* [MDS for neural network output](https://raw.githubusercontent.com/peteris-racinskis/DIA_PD/master/tex/mds-classifier.png)
* [MDS for expected value model output](https://raw.githubusercontent.com/peteris-racinskis/DIA_PD/master/tex/mds-model.png)
* [IsoMDS for neural network output](https://raw.githubusercontent.com/peteris-racinskis/DIA_PD/master/tex/iso-mds-classifier.png)
* [IsoMDS for expected value model output](https://raw.githubusercontent.com/peteris-racinskis/DIA_PD/master/tex/iso-mds-model.png)

## Datasets

* *votes.csv* - the vote component of the publicly available [EuroVision Song Contest Dataset](https://github.com/Spijkervet/eurovision-dataset)

## Scripts

* *datagen.py* - converts vote dataset into tensor form, where every layer is a weighted adjacency matrix of the participating countries. Missing values indicated by -1. Normalizes the score interval values 10, 12 to 9, 10. Actually very poorly implemented but I only needed to run it a few times.
* *vectorize.py *- convert the nation codes into one-hot vector encodings. Optionally apply the same to scores, or leave them as discrete string classes.
* *model.py* - handrolled estimation method. Reduces the data tensor by averaging across competition stages, while keeping track of adjacency counts. Computes average scores, corrects for differences between participant nations, then corrects for biased voting patterns for each voting nation.
* *model-classifier.py*- rather than trying to work with the data directly to estimate the expected values of scores, train a classifier with some more or less randomly discovered parameters to do it instead.
* *mds.R* - MDS, isoMDS visualization of distances. Ripped straight from some other homework assignment of mine.

## Intermediate data files

* *Results-YYYY-YYY*-... - outputs obtained with the tensor-averages-based model.
* *ScoreTensor*-... - precomputed score tensor. Precomputed because it's slow as hell.
* *Vectorized*-... - the one-hot encoded dataset for use with standard scikit-learn classifiers.
* *dumpfile* - contains the neural network weights so I don't have to retrain it every time.
* *classifier*-... - outputs obtained with the neural network model.
