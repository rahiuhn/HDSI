#' Get the performance of HDSI model on any given dataset
#'
#' the functions in this file provide support to HDSI_model.R file
#'
# Perform Lasso, ridge, alasso, aridge
mcoxph=memoise::memoise(survival::coxph)

HDSI_penal_reg=function(df, f, alpha=1, adaptive=0, other_para=para, covariate=c(1), outvar="y"){
  set.seed(1)
  output=other_para['out_type']

  traindf=df
  if(output=="survival"){df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))}
  else{df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c(outvar))}

  if(output=="survival"){variable_list=setdiff(names(df_output[[2]]), c("time", outvar))}
  else{variable_list=setdiff(names(df_output[[2]]), c(outvar))}

  Matrix= model.matrix( ~ ., df_output[[2]][,variable_list])[, -1]

  if(output=="survival"){Y=cbind(time=traindf[,"time"], status=traindf[,outvar])}else{Y=as.matrix(traindf[,outvar])}

  if(output=="survival"){
    if(adaptive==1){
      #Run Ridge
      cv.ridge <- glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="cox")
      ada_coef=coef(cv.ridge, s=cv.ridge$lambda.min)[, 1]
      dif_wt <- 1/abs(matrix(ada_coef))^0.25 ## Using gamma = 1
      dif_wt[dif_wt[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999
      penalty=dif_wt
    }else{penalty=rep(1, length(colnames(Matrix)))}

    fit = glmnet::cv.glmnet(Matrix, Y, alpha=alpha, standardize=F, family="cox", penalty.factor = penalty) # get optimum lambda
    lambda.1se=fit$lambda.1se
    lambda.min=fit$lambda.min

    model = glmnet::glmnet(Matrix, Y, lambda = lambda.1se, alpha=alpha, standardize=F, family="cox", penalty.factor = penalty)
    Coef=as.matrix(coef(model,s=lambda.1se))

    if(length(unique(Coef))==1){
      model = glmnet::glmnet(Matrix, Y, lambda = lambda.min, alpha=alpha, standardize=F, family="cox", penalty.factor = penalty)
      Coef=as.matrix(coef(model,s=lambda.min))
    }
  }else{
    if(adaptive==1){
      #Run Ridge
      cv.ridge <- glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="gaussian")
      dif_wt <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)[2:(ncol(Matrix)+1), 1]))^0.25 ## Using gamma = 1
      dif_wt[dif_wt[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999
      penalty=dif_wt
    }else{penalty=rep(1, length(colnames(Matrix)))}
    fit = glmnet::cv.glmnet(Matrix, Y, alpha=alpha, standardize=F, family="gaussian", penalty.factor = penalty) # get optimum lambda
    lambda.1se=fit$lambda.1se
    lambda.min=fit$lambda.min
    model = glmnet::glmnet(Matrix, Y, lambda = lambda.1se, alpha=alpha, standardize=F, family="gaussian", penalty.factor = penalty)
    Coef=as.matrix(coef(model,s=lambda.1se))
    if(length(unique(Coef))==1){
      model = glmnet::glmnet(Matrix, Y, lambda = lambda.min, alpha=alpha, standardize=F, family="gaussian", penalty.factor = penalty)
      Coef=as.matrix(coef(model,s=lambda.min))
    }
  }
  return(list(model=model,Coef=Coef))
}

HDSI_Lasso = function(df, outvar= "y", f, other_para=para, covariate=c(1)){

  # Define the tuning parameters
  interactionterm= other_para['interactions']
  int_term= other_para['int_term']
  output=other_para['out_type']
  perf_metric= other_para['perf_metric']

  # Run the model
  out_model=HDSI_penal_reg(df=df, f=f, alpha=1, adaptive=0, other_para=other_para, covariate = covariate, outvar=outvar)
  model=out_model$model
  Coef=out_model$Coef

  # Get the performance metric
  if(perf_metric=="beta"){estimate=Coef[,1]}
  else if(perf_metric !="beta" & output=="survival"){
    traindf=df

    df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))
    variable_list=setdiff(names(df_output[[2]]), c("time", outvar))
    Matrix= model.matrix( ~ ., df_output[[2]][,variable_list])[, -1]

    if(length(variable_list)==1){Matrix=cbind(0, Matrix)}
    pred = predict(model,Matrix, type=c("response"))
    Y=cbind(time=traindf[,"time"], status=traindf[,outvar])
    #pred=predict(model, newx=Matrix)
    #estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
    estimate=glmnet::Cindex(pred, Y)
  }
  else{estimate=model$dev.ratio}

  # output model
  df_out=HDSI_output_format(coeflist = rownames(Coef), coefficient = Coef[,1], model_perf=estimate)

  return(list(model, df_out))
}

HDSI_Ridge = function(df, outvar= "y", f, other_para=para, covariate=c(1)){
  # Define the tuning parameters
  interactionterm= other_para['interactions']
  int_term= other_para['int_term']
  output=other_para['out_type']
  perf_metric= other_para['perf_metric']

  # Run the model
  out_model=HDSI_penal_reg(df=df, f=f, alpha=0, adaptive=0, other_para=other_para, covariate = covariate, outvar=outvar)
  model=out_model$model
  Coef=out_model$Coef

  # Get the performance metric
  if(perf_metric=="beta"){estimate=Coef[,1]}
  else if(perf_metric !="beta" & output=="survival"){
    traindf=df

    df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))
    variable_list=setdiff(names(df_output[[2]]), c("time", outvar))
    Matrix= model.matrix( ~ ., df_output[[2]][,variable_list])[, -1]

    if(length(variable_list)==1){Matrix=cbind(0, Matrix)}
    pred = predict(model,Matrix, type=c("response"))
    Y=cbind(time=traindf[,"time"], status=traindf[,outvar])
    #pred=predict(model, newx=Matrix)
    #estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
    estimate=glmnet::Cindex(pred, Y)
  }
  else{estimate=model$dev.ratio}

  # output model
  df_out=HDSI_output_format(coeflist = rownames(Coef), coefficient = Coef[,1], model_perf=estimate)

  return(list(model, df_out))
}

HDSI_Alasso = function(df, outvar= "y", f, other_para=para, covariate=c(1)){
  # Define the tuning parameters
  interactionterm= other_para['interactions']
  int_term= other_para['int_term']
  output=other_para['out_type']
  perf_metric= other_para['perf_metric']

  # Run the model
  out_model=HDSI_penal_reg(df=df, f=f, alpha=1, adaptive=1, other_para=other_para, covariate = covariate, outvar=outvar)
  model=out_model$model
  Coef=out_model$Coef

  # Get the performance metric
  if(perf_metric=="beta"){estimate=Coef[,1]}
  else if(perf_metric !="beta" & output=="survival"){
    traindf=df

    df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))
    variable_list=setdiff(names(df_output[[2]]), c("time", outvar))
    Matrix= model.matrix( ~ ., df_output[[2]][,variable_list])[, -1]

    if(length(variable_list)==1){Matrix=cbind(0, Matrix)}
    pred = predict(model,Matrix, type=c("response"))

    #pred=predict(model, newx=Matrix)
    #estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
    estimate=glmnet::Cindex(pred, Y)
  }
  else{estimate=model$dev.ratio}

  # output model
  df_out=HDSI_output_format(coeflist = rownames(Coef), coefficient = Coef[,1], model_perf=estimate)

  return(list(model, df_out))
}

HDSI_Aridge = function(df, outvar= "y", f, other_para=para, covariate=c(1)){
  # Define the tuning parameters
  interactionterm= other_para['interactions']
  int_term= other_para['int_term']
  output=other_para['out_type']
  perf_metric= other_para['perf_metric']

  # Run the model
  out_model=HDSI_penal_reg(df=df, f=f, alpha=0, adaptive=1, other_para=other_para, covariate = covariate, outvar=outvar)
  model=out_model$model
  Coef=out_model$Coef

  # Get the performance metric
  if(perf_metric=="beta"){estimate=Coef[,1]}
  else if(perf_metric !="beta" & output=="survival"){
    traindf=df

    df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))
    variable_list=setdiff(names(df_output[[2]]), c("time", outvar))
    Matrix= model.matrix( ~ ., df_output[[2]][,variable_list])[, -1]

    if(length(variable_list)==1){Matrix=cbind(0, Matrix)}
    pred = predict(model,Matrix, type=c("response"))
    Y=cbind(time=traindf[,"time"], status=traindf[,outvar])
    #pred=predict(model, newx=Matrix)
    #estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
    estimate=glmnet::Cindex(pred, Y)
  }
  else{estimate=model$dev.ratio}

  # output model
  df_out=HDSI_output_format(coeflist = rownames(Coef), coefficient = Coef[,1], model_perf=estimate)

  return(list(model, df_out))
}

# Perform regression and forward

HDSI_Regression = function(df=inputdf, outvar="y", f, other_para=para, covariate=c(1)){
  # Define the tuning parameters
  perf_metric= other_para['perf_metric']
  interactionterm= other_para['interactions']
  output=other_para['out_type']
  perf_metric= other_para['perf_metric']

  # Define the input and output predictors
  traindf = df
  if(output=="survival"){df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))}
  else{df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c(outvar))}

  traindf= df_output[[2]]

  if(output=="survival"){variable_list=setdiff(names(df_output[[2]]), c("time", outvar))}
  else{variable_list=setdiff(names(df_output[[2]]), c(outvar))}

  if(output=="survival"){Y=cbind(time=traindf[,"time"], status=traindf[,outvar])}else{Y=as.matrix(traindf[,outvar])}

  # Run the model
  if(output=="survival"){
    max_scope=as.formula(paste("survival::Surv(time, status)~", paste(variable_list, collapse = "+")))
    model=tryCatch({survival::coxph(max_scope,data = traindf)}, error=function(e){NULL})}
    #model=tryCatch({mcoxph(max_scope,data = traindf)}, error=function(e){NULL})
    #print(model)}
  else{
    max_scope=as.formula(paste("y~", paste(variable_list, collapse = "+")))
    model=tryCatch({stats::lm(max_scope,data = traindf)}, error=function(e){NULL})}

  # Get the performance metric
  model_summary=summary(model)
  if(perf_metric=="beta"){estimate=model$coefficients}
  else if(perf_metric !="beta" & output=="survival"){
    Matrix=model.matrix(f,df)[,-1]
    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix
    pred = stats::predict(model,newdata = df,times = 5*365.25)
    estimate=glmnet::Cindex(pred, Y)
    estimate=ifelse(estimate<0.5, 1-estimate, estimate)
    # pred=predict(model, newx=Matrix)
    # estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
  }
  else{estimate=model_summary$adj.r.squared}


  # output model
  df_out=HDSI_output_format(coeflist = names(model$coefficients), coefficient = model$coefficients, model_perf=estimate)

  return(list(model, df_out))
}

HDSI_Forward = function(df=inputdf, outvar="y", f, other_para=para, covariate=c(1)){
  # Define the tuning parameters
  perf_metric= other_para['perf_metric']
  interactionterm= other_para['interactions']
  int_term=other_para['int_term']
  output=other_para['out_type']

  # Define the input and output predictors
  traindf=df
  if(output=="survival"){df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c("time", outvar))}
  else{df_output=df_creator(covariate = covariate, df=traindf, f=f, outcome = c(outvar))}
  traindf= df_output[[2]]

  names(traindf)=stringr::str_replace(names(traindf), ":", "")

  if(output=="survival"){variable_list=setdiff(names(traindf), c("time", outvar))}
  else{variable_list=setdiff(names(traindf), c(outvar))}

  if(output=="survival"){Y=cbind(time=traindf[,"time"], status=traindf[,outvar])}else{Y=as.matrix(traindf[,outvar])}

  # Run the model
  if(output=="survival"){
    max_scope=as.formula(paste("survival::Surv(time, status)~", paste(variable_list, collapse = "+")))
    M=survival::coxph(as.formula(paste("survival::Surv(time, status)~", paste(covariate, collapse = "+"))), data = traindf)}
  else{
    max_scope=as.formula(paste("y~", paste(variable_list, collapse = "+")))
    M=stats::lm(as.formula(paste("y~", paste(covariate, collapse = "+"))), data = traindf)}
  model=stats::step(M, direction = "forward", k=log(nrow(traindf)),trace=FALSE, scope=max_scope)

  # Get the performance metric
  model_summary=summary(model)
  if(perf_metric=="beta"){estimate=model$coefficients}
  else if(perf_metric !="beta" & output=="survival"){
    Matrix=model.matrix(f,df)[,-1]
    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix
    pred = stats::predict(model,newdata = df,times = 5*365.25)
    estimate=glmnet::Cindex(pred, Y)
    estimate=ifelse(estimate<0.5, 1-estimate, estimate)
    # pred=predict(model, newx=Matrix)
    # estimate=concordance.index(pred, Y[,1], Y[,2])$c.index #Cindex(pred, Y)
  }
  else{estimate=model_summary$adj.r.squared}

  # output model
  df_out=HDSI_output_format(coeflist = names(model$coefficients), coefficient = model$coefficients, model_perf=estimate)
  full_coef=data.frame(Variable=variable_list, stringsAsFactors = F)
  df_out=merge(df_out, full_coef, by=c("Variable"), all=T)
  df_out[is.na(df_out)] <- 0


  return(list(model, df_out))
}

m_HDSI_Lasso = memoise::memoise(HDSI_Lasso)
m_HDSI_Ridge = memoise::memoise(HDSI_Ridge)
m_HDSI_Alasso = memoise::memoise(HDSI_Alasso)
m_HDSI_Aridge = memoise::memoise(HDSI_Aridge)
m_HDSI_Regression = memoise::memoise(HDSI_Regression)
m_HDSI_Forward = memoise::memoise(HDSI_Forward)
