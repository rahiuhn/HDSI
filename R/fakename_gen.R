#' Create the Fakename and constant names in the model
#'
#' This function takes a dataframe/vector list as an input and create the standard column and row names.
#' The column format is "X(numeric)" like "X1" and "X2". Rows are labelled as 1,2,3,4...
#'
#' @param datafile dataframe or vector list to change the names
#' @param train_test to determine if the data contain separate train and test files
#' @param outcome_var to determine if the dataset has any outcome variable
#' @param num_outcome to detemrine the number of outcome varaibles. It takes variable name as input
#' @return dataframe or vectorlist with changed names as well as list of realnames
#' @export

fakename_gen <- function(datafile, train_test=T, outcome_var=NA, num_outcome=0){
  #"Check if data is split into train and test"
  if(train_test){
    train=tryCatch({datafile[[1]]},
                   error=function(e){print("Provide the 'datafile' in list
                                           format with 1st element as train and 2nd elment as test")})
    test=tryCatch({datafile[[2]]},
                   error=function(e){print("Provide the 'datafile' in list format with 1st element as train and 2nd elment as test")})
  }else{train=datafile; test=NA}

  #str(train)

  # Create Fake rownames
  if(class(train)=="data.frame"){
    original_rows_train=rownames(train)
    rownames(train)=seq(1:nrow(train))
    if(!is.na(test)){
      original_rows_test=rownames(test)
      rownames(test)=seq(1:nrow(test))
    }else{original_rows_test=NA}
  }else{original_rows_train=original_rows_test=NA}

  # Create Fake colnames

  ## Check if the data has an outcome variable/s
  if(!is.na(outcome_var)){noname_change=length(outcome_var)}
  else if (num_outcome>0){noname_change=num_outcome}
  else{noname_change=0}

  ## Store the real col_names
  if(class(train)=="data.frame"){realnames=names(train)}else{realnames=train}

  ## Define the number of outcome variables
  if(is.na(outcome_var)){realnames=realnames[1:(length(realnames)-noname_change)]}
  else{realnames=setdiff(realnames, outcome_var)}

  fakename_col=paste0("X",seq(1:(length(realnames))), "_")

  #str(fakename_col)

  if(class(train)=="data.frame"){
    names(train)[names(train) %in% realnames] = fakename_col
    if(!is.na(test)){names(test)[names(test) %in% realnames] = fakename_col}
  }
  else{
    train[train %in% realnames]=fakename_col
    if(!is.na(test)){test[test %in% realnames] = fakename_col}
  }

  if(train_test){outfile=list(train=train, test=test)}else{outfile=train}
  #str(outfile)

  output=list(realcolnames = realnames,
              real_row_train = original_rows_train,
              real_row_test = original_rows_test,
              original_file = datafile,
              modified_file = outfile)

  return(output)
}
