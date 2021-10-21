#' Get the performance of HDSI model on any given dataset
#'
#' the functions in this file provide support to HDSI_model.R file
#'
#'#' @export
HDSI_formula_gen = function(other_para=para){
  #str(other_para)
  # Define the tuning parameters
  interactionterm= other_para['interactions']
  int_term= as.numeric(other_para['int_term'])
  output=other_para['out_type']
  #str(other_para)
  if(output=="survival"){y="survival::Surv(time,status)"}else{y="y"}

  # Define the input and output predictors
  if(interactionterm==T){
    if(int_term==3){ f=stats::as.formula(paste(y," ~ .*.*.")) }else{ f=stats::as.formula(paste(y," ~ .*.")) }
  }else{f=stats::as.formula(paste(y," ~ ."))}
  return(f)
}

#' @export
df_creator = function(covariate, df, f, outcome="y"){
  if(all(covariate %in% 1) | is.na(covariate)){
    Matrix=stats::model.matrix(f,df)[,-1]
    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix
  }else{
    no_covariate=setdiff(names(df), covariate)

    df_no_covariate=df[,no_covariate]
    Matrix=stats::model.matrix(f,df_no_covariate)[,-1]

    no_covariate_no_outcome=setdiff(no_covariate, outcome)
    df[,no_covariate_no_outcome]=list() # to account for categoricals

    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix
  }
  return(list(mat=Matrix, newdf=df))
}

#' @export
var_organise = function(inputlist, symbol="_"){
  org_var=lapply(inputlist, function(x) {
    a=strsplit(x, symbol);
    ifelse(length(a[[1]])>1, paste0(naturalsort::naturalsort(a[[1]]),collapse = symbol), a[[1]])})
  return(unlist(org_var))
}

#' @export
HDSI_output_format=function(coeflist, coefficient, model_perf){
  if(is.null(coefficient)){
    df_out=data.frame(Variable=NA, n=NA, s=NA, beta=NA, imp= NA, stringsAsFactors = F)
  }else{
    df_out=data.frame(Variable=coeflist, n=1, s=1, beta=model_perf, imp= coefficient, stringsAsFactors = F)
    names(df_out)=c("Variable", "n", "s", "beta", "imp")
    df_out$s[which(df_out$imp==0 | is.na(df_out$imp))]=0
    df_out$beta[which(df_out$imp==0 | is.na(df_out$imp))]=0
  }

  return(df_out)
}

#' @export
HDSI_para_control = function(interactions=T, int_term=2, intercept=T, out_type="continuous",
                             perf_metric=c("beta", "rsq", "rsq_beta")){
  para=c(out_type = out_type, # Is the data of type survival or continuous
         int_term = int_term, # Interaction level to consider 2 or 3
         interactions = interactions, # to study interactions: TRUE OR FALSE
         perf_metric = perf_metric,# number of reduced dimensions ot create for sparse PLS
         intercept = intercept) # to consider intercept during performance evaluation
  return(para)
}

#Summarize the result of each bootstrap
#' @export
f_summary_rsquare=function(x, cint=0.95, min_max=c("ci", "quartile", "min", "only_min")){
  #cat(x)
  quant=(1-cint)/2
  x=x[!is.na(x)]
  M=mean(x, na.rm=T); stdev=sd(x, na.rm=T); Med=median(x, na.rm = T)

  quan=quantile(x, c(quant,1-quant)); lqi=quan[1]; uqi=quan[2]
  #gm=geometric.mean(sqrt(x^2), na.rm = T)
  # Summarize the results based on the criteria: Confidene Interval, Quartile Interval or Minimum and Maximum
  if(min_max=="ci"){
    conf=unlist(list(attributes(suppressMessages(miscset:::confint.numeric(x, na.rm = T, level = cint)))));
    lci=conf[2]; uci=conf[3]; #cat(lci[[1]])
    out=c(mean=M, sd=stdev, median=Med, lowerci=lci, upperci=uci, lowerqi=lqi, upperqi=uqi)
  }
  else if(min_max=="quartile"){
    out = c(mean=M, sd=stdev, median=Med, lowerci=lqi, upperci=uqi, lowerqi=lqi, upperqi=uqi)
  }
  else{
    minimum = min(x, na.rm = T); maximum = max(x, na.rm = T)
    out=c(mean=M, sd=stdev, median=Med, lowerci=minimum, upperci=maximum, lowerqi=lqi, upperqi=uqi)
  }

  blank=c(mean=0, sd=0, median=0, lowerci=0, upperci=0, lowerqi=0, upperqi=0)
  if(any(is.na(out))){out <- blank}
  names(out) = c("mean","sd","median","lowerci","upperci","lowerqi","upperqi")
  #print(out)
  return(out)
}
#' @export
f_summary_beta=function(x, cint=0.95, min_max = c("quartile")){
  quant=(1-cint)/2
  x=x[!is.na(x)]
  M=mean(x, na.rm=T); stdev=sd(x, na.rm=T); Med=median(x, na.rm = T)

  conf=unlist(list(attributes(suppressMessages(miscset:::confint.numeric(x, na.rm = T, level = cint)))));
  lci=conf[2]; uci=conf[3]; #cat(lci[[1]])

  quan=quantile(x, c(quant,1-quant)); lqi=quan[1]; uqi=quan[2]

  if(min_max == "quartile"){
    out=c(mean=M, sd=stdev, median=Med, lowerci=lqi, upperci=uqi, lowerqi=lqi, upperqi=uqi)
  }
  else{
    out=c(mean=M, sd=stdev, median=Med, lowerci=lci, upperci=uci, lowerqi=lqi, upperqi=uqi)
  }

  blank=c(mean=0, sd=0, median=0, lowerci=0, upperci=0, lowerqi=0, upperqi=0)
  if(any(is.na(out))){out<-blank}
  return(out)
}

# Feature selection
#' @export
beta_feature_select = function(df=res_summary){
  df$Selection = apply(df[,-1],1, function(x) {
    range = spatstat.utils::inside.range(0, r=c(min(x[4:7]), max(x[4:7])));
    ifelse(range==TRUE,0,x[1])})
  ## Extract selected features
  cond=df$Variable != "(Intercept)" & df$Selection!=0
  sel_feature = df[cond,]
  return(sel_feature)
}
#' @export
HDSI_var_extract=function(raw_feature_list){
  ## Remove the connectors
  connections=c("_", ":", "'", "`")

  var_extractor=function(x, connections, replacement = rep("_", 4)){
    raw_var=stringi::stri_replace_all_fixed(x, pattern = connections, replacement = replacement, vectorize_all = F)
    split_var = strsplit(raw_var, "_")[[1]]
    clean_varindex=grep("X",split_var)
    feature=split_var[clean_varindex]
    return(feature)
  }

  raw_varlist = lapply(raw_feature_list, function(x) var_extractor(x, connections=connections))
  raw_list=lapply(raw_varlist, function(x)paste(x, "_", sep = ""))

  # Extract all interactions
  var_length = unlist(lapply(raw_varlist,length))
  IV_index = which(var_length >1)
  IV_list = unlist(lapply(raw_varlist[IV_index], function(x) paste(x, collapse = "_")))
  IV_numb=length(IV_list)

  # Extract all marginals
  MV_list=unique(unlist(raw_varlist))
  MV_numb=length(MV_list)

  # Get the final feature list
  sel_feature_list = union(MV_list, IV_list)
  sel_feature_list_nointercept=sel_feature_list[sel_feature_list!="(Intercept)"]
  feature_numb=length(sel_feature_list_nointercept)
  #print(sel_feature_list)

  result= list(features=sel_feature_list_nointercept, feature_number=feature_numb,
               MV_list= MV_list, MV_numb=MV_numb,
               IV_list=IV_list, IV_numb=IV_numb, raw_list=raw_list)
  return(result)
}
#' @export
HDSI_feature_selection=function(approach=c("mp", "beta", "mp_beta"), df, min_max= "min", cint = 0.95, sd_level=1){

  # Select the feature_selection approach
  if(approach == "mp"){
    # Summarize the results
    res_summary = plyr::ddply(df[,c("Variable","beta")], plyr::.(Variable),
                              function(x) {a=f_summary_rsquare(x$beta, min_max = min_max);unlist(a)})

    # Feature Selection
    raw_list=res_summary[, c("Variable", "lowerci", "upperci")]
    Lowercond= mean(raw_list$lowerci , na.rm = T) + sd_level*sd(raw_list$lowerci , na.rm = T)
    uppercond= mean(raw_list$upperci , na.rm = T) + sd_level*sd(raw_list$upperci , na.rm = T)
    sel_feature=raw_list[raw_list$lowerci > Lowercond & raw_list$upperci > uppercond, ]

    features_result=HDSI_var_extract(raw_feature_list = sel_feature$Variable)
    sel_feature_list=features_result$features
    raw_list = features_result$raw_list
  }
  else if (approach == "beta"){
    # Summarize the results
    res_summary = plyr::ddply(df[,c("Variable","imp")], plyr::.(Variable), function(x) {a=f_summary_beta(x$imp, cint = cint); unlist(a)})

    # Feature Selection
    sel_feature=beta_feature_select(res_summary)

    features_result=HDSI_var_extract(raw_feature_list = sel_feature$Variable)
    sel_feature_list=features_result$features
    raw_list = features_result$raw_list
  }
  else{
    # Feature Selection based on Model Performance, i.e. R square or C index

    # Summarize the results and get the Minimum (/lower CI) of each variable R square
    res_summary = plyr::ddply(df[,c("Variable","beta")], plyr::.(Variable), function(x)
                                  {a = f_summary_rsquare(x$beta, min_max = min_max); unlist(a)})

    # Feature Selection based on minimum value
    raw_list = res_summary[, c("Variable", "lowerci", "upperci")]

    ## Define the cutoff
    cond = mean(raw_list$lowerci , na.rm = T) + sd_level*sd(raw_list$lowerci , na.rm = T)

    sel_feature = raw_list[raw_list$lowerci > cond, ]

    if(nrow(sel_feature)>0){
      new_df = df[df$Variable %in% sel_feature$Variable, ]

      # Summarize the results
      res_summary = plyr::ddply(new_df[,c("Variable","imp")], plyr::.(Variable), function(x) {a=f_summary_beta(x$imp, cint = cint); unlist(a)})

      # Feature Selection
      sel_feature = beta_feature_select(res_summary)
      features_result = HDSI_var_extract(raw_feature_list = sel_feature$Variable)
      sel_feature_list = features_result$features
      raw_list = features_result$raw_list
    }else{sel_feature_list=NA; raw_list=NA}
  }

  feature_numb = length(sel_feature_list)
  IV = grep("_", sel_feature_list)
  IV_list = sel_feature_list[IV]
  IV_numb = length(IV_list)

  if(IV_numb>0){MV_list = sel_feature_list[-IV]}else{MV_list = sel_feature_list}

  MV_numb = length(MV_list)
  result= list(sel_feature_list=sel_feature_list, raw_list=raw_list, feature_number=feature_numb,
               MV_list= MV_list, MV_numb=MV_numb, IV_list=IV_list, IV_numb=IV_numb)
  return(result)
}

# Performance evaluation of the model
#' @export
predict_metric=function(actualy=validationdf[,outname[x]], predictedy=y_estimate, varlist=1){

  Correlation = stats::cor(actualy,predictedy, method = "pearson")
  mean_square_error = MLmetrics::MSE(actualy,predictedy)
  RMSE= sqrt(mean_square_error)
  bic=length(actualy)*log(mean_square_error) + 2*log(varlist+1)
  rsquare = Correlation^2
  res=data.frame(Corr=Correlation, RMSE=RMSE, rsquare=rsquare, BIC= bic,stringsAsFactors = F, check.names = T)
  names(res)=c("Corr", "RMSE", "rsquare", "BIC") # Sometimes computer is not adding proper names

  return(res)
}
#' @export
HDSI_performance=function(raw_list, df, outvar, model_tech, covariate = covariate, output="continuous"){
  # Convert raw_list into variable list
  varlist=sapply(raw_list, function(x) ifelse(length(x)>1, paste(x,collapse = "*"),x))
  varlist=union(varlist, covariate[-1])
  # print(varlist)
  if(length(varlist)<1){Perf=data.frame(Corr=0,RMSE=NA, rsquare=0, datatype=NA, stringsAsFactors = F); return(Perf)}
  if(is.na(varlist)){Perf=data.frame(Corr=NA,RMSE=NA, rsquare=NA, stringsAsFactors = F); return(Perf)}
  #print(varlist)
  traindf=df[[1]]
  testdf=df[[2]]
  #str(traindf)
  # Create the formula and model from it
  if(model_tech=="aridge"){
    f=as.formula(paste("~", paste(varlist, collapse = "+")))
    #print(f)
    #print(names(traindf))
    Matrix=model.matrix(f, traindf)[,-1]
    # Define the Y
    if(output=="survival"){Y=cbind(time=traindf[,"time"], status=traindf[,outvar])}else{Y=as.matrix(traindf[,outvar])}
    # Make the model
    if(output=="survival"){
      #Run Ridge for weights
      cv.ridge <- glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="cox")
      ada_coef=coef(cv.ridge, s=cv.ridge$lambda.min)[, 1]
      dif_wt <- 1/abs(matrix(ada_coef))^0.25 ## Using gamma = 1
      dif_wt[dif_wt[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999
      penalty=dif_wt

      #Run Ridge for model
      fit = glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="cox", penalty.factor = penalty) # get optimum lambda
      lambda.1se=fit$lambda.1se
      lambda.min=fit$lambda.min

      model = glmnet::glmnet(Matrix, Y, lambda = lambda.1se, alpha=0, standardize=F, family="cox", penalty.factor = penalty)
      Coef=as.matrix(coef(model,s=lambda.1se))

      if(length(unique(Coef))==1){
        model = glmnet::glmnet(Matrix, Y, lambda = lambda.min, alpha=0, standardize=F, family="cox", penalty.factor = penalty)
        Coef=as.matrix(coef(model,s=lambda.min))
    }
    }
    else{
      #Run Ridge for weights
      cv.ridge <- glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="gaussian")
      ada_coef=coef(cv.ridge, s=cv.ridge$lambda.min)[, 1]
      dif_wt <- 1/abs(matrix(ada_coef))^0.25 ## Using gamma = 1
      dif_wt[dif_wt[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999
      penalty=dif_wt

      #Run Ridge for model
      fit = glmnet::cv.glmnet(Matrix, Y, alpha=0, standardize=F, family="gaussian", penalty.factor = penalty) # get optimum lambda
      lambda.1se=fit$lambda.1se
      lambda.min=fit$lambda.min

      model = glmnet::glmnet(Matrix, Y, lambda = lambda.1se, alpha=0, standardize=F, family="gaussian", penalty.factor = penalty)
      Coef=as.matrix(coef(model,s=lambda.1se))

      if(length(unique(Coef))==1){
        model = glmnet::glmnet(Matrix, Y, lambda = lambda.min, alpha=0, standardize=F, family="gaussian", penalty.factor = penalty)
        Coef=as.matrix(coef(model,s=lambda.min))
      }
    }
  }
  else{
    if(output=="survival"){
    max_scope=as.formula(paste("survival::Surv(time, status)~", paste(varlist, collapse = "+")))
    model=tryCatch({survival::coxph(max_scope,data = traindf)}, error=function(e){NULL})}
    else{
    max_scope=as.formula(paste("y~", paste(varlist, collapse = "+")))
    model=tryCatch({stats::lm(max_scope,data = traindf)}, error=function(e){NULL})}
    # print(model)
  }

  # Estimate the prediction performance
  df_list=list(train = df[[1]], test = df[[2]])
  if(output == "survival"){
    Performance=lapply(1:2, function(x){
      Perf=HDSI_prediction_Survival(model=model, raw_list=varlist, df=df_list[[x]], outvar=outvar, technique = model_tech)
      Perf$datatype=names(df_list)[x]
      #Perf$Corr = stats::extractAIC(model)[2]
      Perf
    })
  }
  else{
    Performance=lapply(1:2, function(x){
      Perf=HDSI_prediction_continuous(model=model, raw_list=varlist, df=df_list[[x]], outvar=outvar, technique = model_tech)
      Perf$datatype=names(df_list)[x]
      Perf
    })}

  Concat_perf=do.call(rbind,Performance)
  return(Concat_perf)
  #return(model)
}

#' @export
HDSI_prediction_continuous=function(model, raw_list, df, outvar, technique){
  # Define the variables
  variable_list = raw_list[raw_list != "(Intercept)" ]
  outcome = outvar

  # Perform prediction using the model
  if(length(variable_list)<1){Perf=data.frame(Corr=NA,RMSE=NA, rsquare=NA, stringsAsFactors = F); return(Perf)}

  if(is.na(variable_list)){Perf=data.frame(Corr=NA,RMSE=NA, rsquare=NA, stringsAsFactors = F); return(Perf)}

  #f=HDSI_formula_gen(other_para=other_para)
  f=as.formula(paste("~", paste(variable_list, collapse = "+")))
  Matrix=model.matrix(f, df)[,-1]

  if (technique=="reg" | technique=="Forward" | technique=="forward" | technique=="Reg"){
    #Matrix=model.matrix(f,df)[,-1]
    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix

    pred = stats::predict(model,df)
  }
  else {
    # traindf=df
    #
    # f=as.formula(paste("~", paste(variable_list, collapse = "+")))
    # Matrix=model.matrix(f, traindf)[,-1]
    if(length(variable_list)==1 & ncol(Matrix) == 1){Matrix=cbind(0, Matrix)}
    # str(Matrix)
    pred=predict(model,Matrix)
  }

  ## Estimate the performance
  Perf=predict_metric(actualy = df[, outcome], predictedy = pred)[,1:3]
  return(Perf)
}
#' @export
HDSI_prediction_Survival=function(model, raw_list, df, outvar, technique){
  variable_list = raw_list[raw_list != "(Intercept)" ]

  # Perform prediction using the model
  if(length(variable_list)<1){Perf=data.frame(Corr=0,RMSE=NA, rsquare=0, stringsAsFactors = F); return(Perf)}

  if(is.na(variable_list)){Perf=data.frame(Corr=0,RMSE=NA, rsquare=0, stringsAsFactors = F); return(Perf)}

  f=as.formula(paste("~", paste(variable_list, collapse = "+")))
  Matrix=model.matrix(f, df)[,-1]

  Y=cbind(time=df[,"time"], status=df[,outvar])

  if(technique=="reg" | technique=="Forward" | technique=="forward" | technique=="Reg"){
    #Matrix=model.matrix(f,df)[,-1]
    df[,colnames(Matrix)]=list()
    df[,colnames(Matrix)]=Matrix
    pred = stats::predict(model,newdata = df,times = 5*365.25)
  }
  else{
    #traindf=df

    # f=as.formula(paste("~", paste(variable_list, collapse = "+")))
    # Matrix=model.matrix(f, traindf)[,-1]

    #print("This is the Tag")

    if(length(variable_list)==1){Matrix=cbind(0, Matrix)}
    pred = predict(model,Matrix, type=c("response"))
  }

  ## Estimate the performance
  c_value=glmnet::Cindex(pred, Y)
  c_value=ifelse(c_value<0.5, 1-c_value, c_value)
  Perf=data.frame(Corr=stats::extractAIC(model)[2], RMSE=c_value, rsquare=0, stringsAsFactors = F)

  return(Perf)
}

#' @export
feat_freq = function(featurelist, trials = 10){
  feat = unlist(featurelist)
  ffreq = table(feat)/trials
  return(ffreq)
}
