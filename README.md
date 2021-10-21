# High Dimensional Selection with Interactions (HDSI): A short tutorial
High Dimensional Selection with Interactions (HDSI) algorithm allows feature selection in high dimensional data while incorporating interaction terms. It is an ensemble approach (https://doi.org/10.1371/journal.pone.0246159).

This tutorial will explain how to run the HDSI code. The current HDSI algorithm can provide 
1) list of selected marginal features. 
2) list of interaction terms.
3) Predictive performance of selected features both in training and test dataset. The current HDSI algorithm allowes use of regression and penalized adaptive ridge regression to build the predictive model.
 

## Installing the packages
This algorithm uses many R packages which needs to be installed or loaded before using this algorithm. 
```
# Versions used in the current algorithm are:
# memoise (>= 1.1.0), dplyr (>= 0.8.4), stringr (>= 1.4.0), stats (>= 3.6.2), scales (>= 1.1.0), caret (>= 6.0-86), 
magrittr (>= 1.5), janitor (>= 1.2.1), MLmetrics (>= 1.1.1), mice (>= 3.8.0), GA (>= 3.2), MCMCpack (>= 1.4-8), 
copula (>= 1.0-0), gtools (>= 3.8.2), mosaic (>= 1.8.2), future (>= 1.20.1), future.apply (>= 1.6.0), missForest (>= 1.4), 
pkgcond (>= 0.1.0), bnstruct (>= 1.0.8), DMwR (>= 0.4.1), VIM (>= 6.1.0), SimMultiCorrData (>= 0.2.2), pbapply (>= 1.4-3)

install.packages("pacman")
library(pacman)
pacman::p_load(memoise, dplyr, stringr, stats, scales, caret, magrittr, janitor, MLmetrics, mice, GA, MCMCpack, copula, 
gtools, mosaic, future, future.apply, missforest, pkgcond, bnstruct, DMwR, VIM, SimMultiCorrData, pbapply)
```

## Generate simulated dataset
The algorithm allows the user to generate an artificial dataset but with limited functionality. The dataset generated contains 20 input features ```varnum```. 80% of missing values ```maxmiss_per``` in each of the 20 features are missing completely at random. The correlation among 20 features lies in the range of [-0.5,0.5]. Currently, only first three features are allowed to have effect on the model. The coefficient values are (0.2, 0.3, 0.4) with interaction coefficient of 10. The training dataset generated has no complete rows (NCR).
```
dataset = data_sim(varnum =20, # 20 features 
                   maxmiss_per= 0.8, # 80% missing per column 
                   corr_seed=1, # seed number to ensure reproducibility
                   testsize = 1050, # sample size of test data
                   samplesize=4200, # total number of samples generated training +test samples
                   effect="Mar", # Only marginal effects are present in the original model. 
                   max_corr= 0.5) # Correlation among features will lie in between [-0.5, 0.5]
traindata = dataset$train
testdata = dataset$test
truetrain = dataset$truedata

write.csv(traindata,"traindata_SCR_Art.csv")
write.csv(testdata,"testdata_SCR_Art.csv")
write.csv(truetrain,"originaldata_SCR_Art.csv")
```

## Run DMU and compare its performance
Once the training and test dataset are obtained. One can run the DMU algorithm. The function ```sim_fit``` allows the user to run and compare 6 missing imputation methods namely DMU (```DMU```), mean imputation (```mean```), kNN impute (```knn```), Random Forest impute (```rf```), complete case analysis (```reg```), mice (only pmm) (```mice```). Additionally, it provides the true model ```true``` if ```truetrain``` variable contains complete data. In case we want to add some complete rows to training data to change it from NCR to SCR (Some complete rows), ```datatype``` should be assigned value ```"createSCR"``` which will randomly transfer 50 samples with complete information from test data to training data. Otherwise, give ```datatype``` suitable value like ```"NCR"``` or ```"SCR"```. 
```
# Get Parameters
clustersize= 4 # number of small complete dataset that will created for DMU
sample_to_featureratio = 2 # Minimum sample size to feature ratio to be maintained in small complete datasets
para= list(seed=1, sample_clustersize = clustersize, sample_to_featureratio=sample_to_featureratio, datatype="createSCR")

## Results
res = sim_fit(x = para, traindata= traindata, testdata = testdata, truetrain = truetrain, 
              technique= c("DMU", "knn", "mice", "reg","mean", "rf"), splitype = "old")
res$datatype = "SCR"
finalres = res
finalres
Output:
            MSE         Approach seed datatype
DMU  0.10094285        beta_Prop    1      SCR
knn  0.09526723         beta_knn    1      SCR
mice 0.17535876 beta_Imputed_Reg    1      SCR
reg  0.09995927         beta_Reg    1      SCR
mean 0.11734702        beta_mean    1      SCR
rf   0.10774280          beta_Rf    1      SCR
true 0.06532334        beta_true    1      SCR
```

## Hyperparameter optimization
In case, the number of small datasets, k ```clustersize``` is not known. Algorithm can provide its optimal value using genetic algorithm based hyperparameter optimization function ```hyperpara-optimize```.
```
clustersize = hyperpara_optimize(optimtype = "opt_GA", seed=1, traindata= traindata, testdata = testdata, datatype = "createSCR")
clustersize
Output: 4
```

## Limitations of Algorithm
1) This algorithm can only process continuous data
2) Currently, output is only the predictive performance in three metrics namely correlation, mean square error and r-squared
3) It doe not directly provide the final model or feature weights. However, one can determine the weights by looking at function ```DMU```. The result is stored in the data.frame ```bay_reg```.
