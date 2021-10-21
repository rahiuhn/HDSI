#' The functionS in this file prepare the dataset for the modeling
#'
#'
#'@export
# Generate the artificial dataset
dataset=function(varnum, setting="No_Correlation", var=c("Mar", "No_Mar", "No_Var"), seed=2, main_var=10, var_effect=0.5, correlation_var=15, correlation_val=5, high_dim=T, train_sample=500){

  # Create the Covariance Matrix
  Sigma=matrix(rep(0,varnum), nrow=varnum, ncol=varnum, byrow=F)
  if(high_dim){
    for(i in 1:varnum){
      for(j in 1:varnum){
        if(i==j){Sigma[i,j]=10}
        else if(i<=correlation_var & j<=correlation_var & setting == "Correlation"){Sigma[i,j]=Sigma[j,i]=correlation_val}
        else{Sigma[i,j]=Sigma[j,i]=1}
      }
    }
  }
  else{
    for(i in 1:varnum){Sigma[i,i]=10}
    # Correlation Settings
    if(setting=="Correlation"){
      Sigma[1,2]=3;Sigma[1,3]=3;Sigma[1,4]=6;Sigma[1,5]=6
      Sigma[2,1]=3;Sigma[3,1]=3;Sigma[4,1]=6;Sigma[5,1]=6
      Sigma[2,3]=3;Sigma[2,4]=2;Sigma[2,5]=1
      Sigma[3,2]=3;Sigma[4,2]=2;Sigma[5,2]=1
      Sigma[3,4]=2;Sigma[3,5]=1
      Sigma[4,3]=2;Sigma[5,3]=1
      Sigma[4,5]=1
      Sigma[5,4]=1
    }
  }

  # Create the input dataset
  #print(Sigma)
  set.seed(seed)
  ta=data.frame(MASS::mvrnorm(n = train_sample+500, rep(0, varnum), Sigma/10))
  variablelist=list()
  for(i in 1:varnum){
    variablelist[[i]]=gsub(" ", "",paste("X",i))
    ta[,i]=mosaic::zscore(ta[,i])
  }
  variablelist=unlist(variablelist)

  # Create the outcome Variable
  intercept=1
  if(high_dim){


    # Generate theinteraction term
    mar_var= paste(c(names(ta)[1:main_var]), collapse = "+")
    int_var= paste(names(ta)[1:(main_var-1)], "*" , names(ta)[2:main_var], collapse = " + ")
    int_var2 = paste(names(ta)[1:(main_var-2)], "*" , names(ta)[3:main_var], collapse = " + ")
    int_var3 = paste(names(ta)[1:(main_var-3)], "*" , names(ta)[4:main_var], collapse = " + ")
    int_var4 = paste(names(ta)[1:(main_var-4)], "*" , names(ta)[5:main_var], collapse = " + ")
    int_var5 = paste(names(ta)[1:(main_var-5)], "*" , names(ta)[6:main_var], collapse = " + ")
    int_var6 = paste(names(ta)[1:(main_var-6)], "*" , names(ta)[7:main_var], collapse = " + ")

    # featlist = list(marfeat, intfeat, intfeat2, intfeat3, intfeat4,intfeat5, intfeat6)

    # f = as.formula(paste("~",mar_var ," +" , int_var))
    f = as.formula(paste("~",mar_var ," +" , int_var," +" , int_var2," +" , int_var3," +" , int_var4," +" , int_var5," +" , int_var6))
    # print(f)
    main_mat = model.matrix(f, ta)
    # str(main_mat)
    # print(main_mat[1:5,])

    # betas = c(rep(var_effect,(2*main_var)-1))
    # beta_value = betas[1:((2*main_var)-1)]
    #
    # if(var=="Mar"){beta_value=beta_value*rep(1,(2*main_var)-1)}
    betas = c(rep(var_effect,ncol(main_mat)-1))
    beta_value = betas[1:(ncol(main_mat)-1)]

    if(var=="Mar"){beta_value=beta_value*rep(1,ncol(main_mat)-1)}
    else if(var=="No_Mar"){beta_value=beta_value*c(rep(0,2), rep(1,main_var-2), rep(1,main_var-1))}
    else if(var=="No_Var"){beta_value=beta_value*c(rep(0,main_var), rep(0,main_var-1))}
    else if(var=="only_int"){beta_value=beta_value*c(rep(0,main_var), rep(1,main_var-1))}
    else {beta_value = beta_value*c(rep(1,main_var), rep(0,main_var-1))}

    # Get Outcome
    set.seed(2)
    random_value = rnorm(n=train_sample+500, mean=0, sd=0.25)
    # coef_value = apply(main_mat,1, function(x) {sum(x*c(intercept, beta_value))})

    ta$y = main_mat %*% c(intercept, beta_value) + random_value
    # ta$y  = coef_value + random_value
  }
  else{
    if(var=="Mar"){beta_a=1; beta_b=1}
    else if(var=="No_Mar"){beta_a=0; beta_b=1}
    else{beta_a=0; beta_b=0}
    b1=0.2*beta_a
    b2=0.3*beta_a
    b3=0.4*beta_b
    b4=0.3*beta_b
    set.seed(2)
    ta$y= intercept + (b1*ta$X1) + (b2*ta$X2) + (b3*ta$X3) + (b4*ta$X1*ta$X2) + rnorm(n=train_sample+500, mean=0, sd=0.25)
  }
  set.seed(2)
  index=sample(1:nrow(ta), train_sample, replace = F)
  traindf=ta[index,]
  validationdf=ta[-index,]
  return(list(train=traindf,test=validationdf))
}
