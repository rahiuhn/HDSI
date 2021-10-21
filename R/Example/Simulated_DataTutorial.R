# Generate Artificial Dataset
df = dataset(varnum =25, setting="Correlation", var=c("Mar"), seed=2, high_dim=F, train_sample=500)

# Perform hyperaparameter optimization
## Define the parameters
HDSI_para = list(model="reg", covariate=c(1), outvar="y", bootstrap=T, effectsize=5, min_max= "min", model_tech ="reg", interactions=T, int_term=2, intercept=T, out_type="continuous", perf_metric=c("mp_beta"))

## Get optimal hyperparameters
library(future.apply)
plan(multisession(workers =10))
opt_para = cv_hyperopt(seeder=1,df=df, sp = HDSI_para)

# Run HDSI
q = floor(opt_para[1])
minr2 = round(opt_para[2],3)
qthresh = round(opt_para[3],3)

optres = HDSI_model(model="reg", inputdf=df , seed=1, bootstrap=T, effectsize=32,
                    k=q,  cint = qthresh, sd_level=minr2, para=HDSI_para_control(interactions=T, int_term=2, intercept=T, out_type="continuous", perf_metric=c("mp_beta")), covariate=c(1),  min_max= c("min"))
