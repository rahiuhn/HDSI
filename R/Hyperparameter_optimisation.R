#' Optimise the hyperparameters
#'
#' Genetic Algorithm is applied to perform the hyperparameter optimisation.

quadraticRoots <- function(a, b, c) {

  #print(paste0("You have chosen the quadratic equation ", a, "x^2 + ", b, "x + ", c, "."))

  discriminant <- (b^2) - (4*a*c)

  if(discriminant < 0) {
    return(paste0("This quadratic equation has no real numbered roots."))
  }
  else if(discriminant > 0) {
    x_int_plus <- (-b + sqrt(discriminant)) / (2*a)
    x_int_neg <- (-b - sqrt(discriminant)) / (2*a)
    # paste0("The two x-intercepts for the quadratic equation are ",
    #        format(round(x_int_plus, 5), nsmall = 5), " and ",
    #        format(round(x_int_neg, 5), nsmall = 5), ".")
    return(x_int_plus)
  }
  else #discriminant = 0  case
    x_int <- (-b) / (2*a)
  #paste0("The quadratic equation has only one root. This root is ", x_int)
  return(x_int)
}

CV_creator = function(input_data = traindf){
  ## Don't use Future and Warm_start simultaneously
  samples = nrow(input_data)
  Warm_start= NULL
  # 5 fold OOB for 5 trials
  ## Create Five trials
  trials=lapply(1:3, function(x) {set.seed(x); fold= caret::createFolds(1:samples, k=5)})

  trial_list = purrr::flatten(trials)

  return(trial_list)
}

#' @export
cv_hyperopt=function(seeder,df, sp = para, Ga_suggest=NA){
  # Optimize the Hyperparameters
  df_p = df
  rownames(df_p[[1]]) = 1:nrow(df_p[[1]])
  ga_fit = function(x){
    # Hyperparameters k, sd_level, beta_quantile
    {
      model = sp[['model']]
      covariate = sp[['covariate']]
      outvar = sp[['outvar']]
      bootstrap = sp[['bootstrap']]
      effectsize = sp[['effectsize']]
      k = floor(x[1])
      min_max = sp[['min_max']]
      cint = round(x[3],3)
      sd_level = round(x[2],3)
      model_tech = sp[['model_tech']]
      interactions = sp[['interactions']]
      int_term = sp[['int_term']]
      intercept = sp[['intercept']]
      out_type = sp[['out_type']]
      perf_metric = sp[['perf_metric']]
    }
    para=HDSI_para_control(interactions=interactions, int_term=int_term, intercept=intercept,
                           out_type=out_type, perf_metric=perf_metric)
    # Create the Cross-Validation
    cv_list = CV_creator(input_data = df_p[[1]])
    # Get Performance from each CV
    predicted_value = future.apply::future_lapply(cv_list, function(x)
    {
      df = list(df_p[[1]][-x,],df_p[[1]][x,])

      # Run the model
      res= HDSI_model(model=model, inputdf=df , covariate=covariate, outvar=outvar, seed=seeder, bootstrap=bootstrap, effectsize=effectsize, k=k, min_max= min_max, cint = cint, sd_level= sd_level, model_tech = model_tech, para= para)

      # Generate Output: RMSE
      output = res$performance
      loss = output$RMSE[output$datatype == "test"]
      saturated_model=lm(y~.*., data = df[[1]])
      sat_y_pred = predict(saturated_model, df[[2]][,-ncol(df[[2]])])
      sat_mse = MLmetrics::MSE(sat_y_pred, df[[2]][,"y"])
      GMC_EST = (((loss^2)*(100))/sat_mse) + ((output$MV[output$datatype == "test"]+output$IV[output$datatype == "test"]+1)*log(100))
      GMC_EST

    },future.seed = T) # ,future.seed = T
    # Pool Performance from each CV
    if(any(is.na(unlist(predicted_value))) | any(is.nan(unlist(predicted_value))) | length(unlist(predicted_value)) == 0){fitness = 1e40}
    else{fitness = mean(unlist(predicted_value), na.rm = T)}
    return(-fitness)
  }

  lb = c(5,-3, 0.8) # k, mp, fp
  max_k = quadraticRoots(1,1,-2*(0.8*nrow(df_p[[1]])-1)) # Find k which consumes almost all the degree of freedom

  ub = c(min(max_k, ncol(df_p[[1]])-2),3,1) # k, mp, fp

  if(is.na(Ga_suggest)){
    GA = GA::ga(type = "real-valued", fitness = ga_fit, lower = lb, upper = ub, popSize = 20, pcrossover = 0.8, pmutation = 0.4, maxiter = 50, seed=3, parallel = F, run = 10, monitor = T)
  }
  else{
    GA = GA::ga(type = "real-valued", fitness = ga_fit, lower = lb, upper = ub, popSize = 20, pcrossover = 0.8, pmutation = 0.4, maxiter = 50, seed=3, parallel = F, run = 10, monitor = F, suggestions = Ga_suggest) #, , suggestions = c(5, 1.5, 0.95)
  }

  bestpara = GA@solution[1,]
  return(bestpara)
  # print(GA@solution)
  # return(GA@solution)
}

