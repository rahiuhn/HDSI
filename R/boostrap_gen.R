#' Prepare the bootstraps for any dataset
#'
#' The functions takes a dataframe/vector list as an input and perform the bootstrapping of rows
#' and sampling of the columns
#'
#' @param p number of features
#' @param k number of features in a sample
#' @param rows number of rows of the dataframe
#' @param interaction_numb level of interactions considered
#' @param effectsize The expected difference between non-zero beta coefficient and zero value coefficient
#' @param feature_name is the list of features which will be sampled in each bootstrap
#' @param inputdf dataframe of the input data
#' @param type is the datatype which can be continuous or survival
#' @param seed_multiplier seed value to ensure result reproducibility
#' @param bootstrap to determine if the sampling is with or without replacement for rows
#' @return vectorlist of all the bootsamples of rows and columns

#' @export
p_est=function(p,k=3,rows=NA, interaction_numb=2, effectsize= c("large", "medium", "small")){

  denominator=choose(p,interaction_numb)
  numerator=choose(k,interaction_numb) # where, 2 is the order of interaction eg of 2 order interaction is X1_X2
  init_prob=numerator/denominator

  # Minimum bootstraps which will give minimum 13 occurrences (consider large effect) of an interaction variable with 99% confidence
  if(effectsize=="large"){prefer_occurence = 13}
  else if (effectsize == "medium"){prefer_occurence = 32}
  else if(effectsize == "small"){prefer_occurence = 200}
  else {prefer_occurence = effectsize}
  # print(c(prefer_occurence,init_prob))
  min_boot= ceiling(prefer_occurence/init_prob) # where 32 is minimum number of occurence desired for an interaction variable during bootstrapping.
  #print(min_boot)
  max_boot= 8*min_boot
  #print(max_boot)
  f_opt=function(x,y=init_prob,max_x=max_boot,prefer=prefer_occurence){
    Val=qbinom(0.05, floor(x), y)
    value=(prefer-Val)^2+(x/max_x)
    return(value)
  }
  bootvalue = optimize(f_opt,interval = c(min_boot,max_boot))
  # print(ceiling(bootvalue$minimum))
  return(ceiling(bootvalue$minimum))
}

#' @export
bootsample=function(p,k=3,rows=NA, interaction_numb=2, effectsize="large", feature_name, inputdf, type=c("continuous", "survival"), seed_multiplier=1, bootstrap=T){
  # Find the number of sample that need to be created
  boots=p_est(p=p,k=k, rows=rows, interaction_numb=interaction_numb, effectsize=effectsize)

  ## Create Variable list for each bootstrap
  if(type=="survival"){
    res=bootsample_survival(boots=boots, features=feature_name, rows=rows, k=k, seed_multiplier = seed_multiplier, inputdf = inputdf, bootstrap = bootstrap)
  }
  else{res=bootsample_continuous(boots=boots, features=feature_name, rows=rows, k=k, seed_multiplier = seed_multiplier, inputdf = inputdf, bootstrap = bootstrap)}

  return(res)
}

#' @export
mbootsample = memoise::memoise(bootsample)
#' @export
bootsample_continuous=function(boots=boots, features, rows, k, seed_multiplier=1, inputdf, bootstrap=T){
  ## Create Variable list for each bootstrap
  features=features
  samples=1:rows
  op <- pbapply::pboptions(type = "timer")
  res=lapply(1:boots, function(x) {
    set.seed(x*seed_multiplier)
    ### Create variable combinations
    random.features <- sample(features, k, replace = FALSE) # replaced samples with features/3
    feature <- naturalsort::naturalsort(random.features)
    ## Create Sample list for each bootstrap
    random.samples=sample(rownames(inputdf), nrow(inputdf), replace=bootstrap)
    list(feature, random.samples)
  })
  pbapply::pboptions(op)
  return(res)
}

#' @export
bootsample_survival=function(boots=boots, features, rows, k, seed_multiplier=1, inputdf, bootstrap=T){
  ## Create Variable list for each bootstrap
  features=features
  samples=1:rows
  #op <- pbapply::pboptions(type = "timer")
  res=lapply(1:boots, function(x){
    #print(x*seed_multiplier)
    set.seed(x*seed_multiplier)
    ### Create variable combinations
    random.features <- sample(features, k, replace = FALSE) # replaced samples with features/3
    #print(random.features)
    feature=naturalsort::naturalsort(random.features)

    ## Create Sample list for each bootstrap
    SURV_1 <- inputdf[which(inputdf$status==1),]
    S_1 <- sample(rownames(inputdf[which(inputdf$status==1),]), nrow(SURV_1), replace=T)
    SURV_0 <- inputdf[which(inputdf$status==0),]
    S_0 <- sample(rownames(inputdf[which(inputdf$status==0),]), nrow(SURV_0), replace=T)
    random.samples <- as.numeric(c(S_1,S_0))
    list(feature, random.samples)
    }) #, cl=20L
  #pbapply::pboptions(op)
  return(res)
}
