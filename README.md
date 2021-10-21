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
# data.table (>= 1.9.4), stats (>= 3.6.2), glmnet (>= 3.0-2), survival (>= 3.1-8), glinternet (>= 1.0.10), pbapply (>= 1.4-2),
  spls (>= 2.2-3), naturalsort (>= 0.1.3), gtools (>= 3.8.1), MLmetrics (>= 1.1.1), MASS (>= 7.3-51.5), mosaic (>= 1.5.0),
  plyr (>= 1.8.5), miscset (>= 1.1.0), spatstat.utils (>= 1.17-0), memoise (>= 1.1.0), dplyr (>= 0.8.4), stringr (>= 1.4.0),
  future.apply (>= 1.6.0), GA (>= 3.2), caret (>= 6.0-86), purrr (>= 0.3.4), ggplot2 (>= 3.3.2), ggpubr (>= 0.4.0)

install.packages("pacman")
library(pacman)
pacman::p_load(data.table, stats, glmnet, survival, glinternet, pbapply, spls, scales, naturalsort, gtools, MLmetrics, MASS, 
mosaic, plyr, miscset, spatstat.utils, memoise, dplyr, stringr, future.apply, GA, caret, purrr, ggplot2, ggpubr)
```

## Generate simulated dataset
The algorithm allows the user to generate an artificial dataset but with limited functionality. The dataset generated contains 25 input features ```varnum```. The coefficient values are (X1 = 0.2, X2 = 0.3, X3 = 0.4, X1X2 = 0.4) with intercept coefficient of 1.
```
df = dataset(varnum =25, # Marginal feature space, p
             setting="Correlation", # Correlation is present among some features
             var=c("Mar"), # Outcome is influenced by both marginal and interaction terms
             seed=2, # seed number to ensure reproducibility
             high_dim=F, # More complex dataset is not allowed 
             train_sample=500) # sample size training data. Test dataset is fixed at 500 samples
             
df # list containing training dataset and test dataset

```

## Run HDSI
Once the training and test dataset are obtained, HDSI algorithm ```HDSI_model``` can do feature selection. It requires information on three hyperparameters, namely, number of features in a sample (q) ```q, k```, coefficient estimate quantile threshold (Qi) ```qthresh, cint``` and minimum R2 threshold (Rf) ```minr2 , sd_level```. HDSI_model can be run for many statistical methods like lasso, adaptive lasso, ridge, adaptive ridge, simple regression, forward regression ```model```. One can increase the number of bootstraps by increasing ```effectsize```. The algorithm allows to add control features ```covariate```, when no covariate is added it should be given the value of 1. Some other parameters ```para``` are also defined regarding the model like level of interactions ```int_term``` and use of intercept in final model ```intercept```.
```
# Get Parameters
q = 
minr2 = 
qthresh = 

optres = HDSI_model(model="reg", inputdf=df , seed=1, effectsize=32,
                    k=q,  cint = qthresh, sd_level=minr2,
                    para=HDSI_para_control(interactions=T, int_term=2, intercept=T,
                                           out_type="continuous", perf_metric=c("mp_beta")),
                    covariate=c(1),  min_max= c("min"))
optres
```
The outcome ```optres``` provide multiple outcomes. Some of the main results are as follows:
1) use ```optres$performance``` to check for model performance in training and testing data
2) use ```optres$fulldata[[2]]$feature``` to check the selected features
3) use ```optres$realname``` to get the real names of the features
4) use ```optres$fakename``` to get the names used by the model. They are the names which are displayed in ```optres$fulldata[[2]]$feature```

## Hyperparameter optimization
In case, three hyperparameters, namely, number of features in a sample (q) ```q, k```, coefficient estimate quantile threshold (Qi) ```qthresh, cint``` and minimum R2 threshold (Rf) ```minr2 , sd_level``` are not known. Algorithm can provide its optimal value using genetic algorithm based hyperparameter optimization function ```cv_hyperopt```.
```
## Define the parameters
HDSI_para = list(model="reg", covariate=c(1), outvar="y", bootstrap=T, effectsize=5, min_max= "min", model_tech ="reg", interactions=T, int_term=2, intercept=T, out_type="continuous", perf_metric=c("mp_beta"))

## Run optimization
plan(multisession(workers =10)) # for parallel computing
opt_para = cv_hyperopt(seeder=1,df=df, sp = HDSI_para)

# Get hyperparameters values
q = floor(opt_para[1])
minr2 = round(opt_para[2],3)
qthresh = round(opt_para[3],3)
```

## Limitations of Algorithm
1) This algorithm can only process continuous data
2) This algorithm is tested only for continuous outcome
3) Outcome feature should be labeled "y" and must be present as the last column in the dataset.
