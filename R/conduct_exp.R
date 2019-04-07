library("dplyr")
library("foreach")
library("data.table")

args = commandArgs(TRUE)
test_type = as.character(args[1])
visit_type = as.character(args[2])
num_lag = as.numeric(args[3])  # Number of lag columns to include, if lag > 10, then it refers to DMM hidden dimension: 12: 2-dim, 13: 3-dim
include_lab = as.logical(args[4]) # Include lab measurement?
include_ethdon = as.logical(args[5]) # Include patient's ethnicity and race, include details of donor info?
eq_train_ratio = as.logical(args[6]) # Train on 50:50 ratio
output = as.character(args[7])
num_folds = as.numeric(args[8]) # Number of cross validation folds
train_thresh_year = as.numeric(args[9]) # Only use data transplant_year >= train_thresh_year 
cutoff_visit = as.logical(args[10]) # For training data, only use visits before 2011

commandArgs = function(...) c(include_lab, include_ethdon, num_lag, eq_train_ratio, train_thresh_year, cutoff_visit)
source("./R/prep_exp.R")

# Columns used
if (is_survival(test_type)) {
  if (grepl("time", test_type)) {
    y_col_train = c("time_since_transplant", "time_next_followup", "diab_in_1_year") # for coxphtime, the is_event status is whether they have diabetes at time of next followup
    y_col_test = y_col_train
  } else if (visit_type != "last") { 
    y_col_train = c("time_to_diab", "is_diab")
    y_col_test = y_col_train
  } else { # otherwise, for last visit, use the current time (it doesn't make sense to use time_to_diab since it will always be <=1 ) and whether they have diabetes now
    y_col_train = c("time_since_transplant", "is_diab")
    y_col_test = c("time_to_diab", "is_diab")
  }
} else {
  y_col_train = "diab_in_1_year"
  y_col_test = y_col_train
}

# Include time since transplant except for cases when y includes time_since_transplant 
# and when we're not looking at first visit only (since first visit's time_since_transplant == 0)
if (!("time_since_transplant" %in% cols) & !("time_since_transplant" %in% y_col_train)  
    & !("time_since_transplant" %in% y_col_test) & !grepl("first", visit_type)) {
  cols = c(cols, "time_since_transplant")
}

nontest_ids = select(df_nontest, TRR_ID, is_diab) %>% unique %>% .$TRR_ID
nontest_y = select(df_nontest, TRR_ID, is_diab) %>% unique %>% .$is_diab
require("caret")
folds = createFolds(nontest_y, num_folds, F)
# Some additional bookkeeping for rf stuffs
rf_nds = c()
rf_val_perfs = c()

result = foreach (i = c(1:num_folds, 0), .combine = rbind) %do% {
  if (i > 0) {
    train_ids = nontest_ids[folds != i]
    val_ids = setdiff(nontest_ids, train_ids)
    test = filter(df_nontest, TRR_ID %in% val_ids)
  } else {
    train_ids = nontest_ids
    test = df_test
  }
 
  
  if (visit_type == "first") {
    if (train_thresh_year == 1990) {
      train = filter(df_nontest, TRR_ID %in% train_ids, 
                     (transplant_year < 2000 & time_since_transplant == 0.5) | 
                       (transplant_year >= 2000 & time_since_transplant == 0))
    } else {
      train = filter(df_nontest, TRR_ID %in% train_ids, time_since_transplant == 0)
    }
  } else if (visit_type == "last") {
    # Since we ignored observations after diab event or the last followup time, we can do this
    train = filter(df_nontest, TRR_ID %in% train_ids) %>% group_by(TRR_ID) %>% 
      filter(time_since_transplant == max(time_since_transplant), time_since_transplant > 0) %>% as.data.frame()
    
  } else if (visit_type == "random") {
    train = filter(df_nontest, TRR_ID %in% train_ids) %>% group_by(TRR_ID) %>% filter(time_since_transplant > 0) %>% 
      sample_n(1) %>% as.data.frame()
  } else if (visit_type == "all") {
    train = filter(df_nontest, TRR_ID %in% train_ids)
  }
  
  if (test_type == "glmnetcox" | grepl("aft", test_type)) {
    train = filter(train, time_to_diab > 0)
    test = filter(test, time_to_diab > 0)
  }
  
  if (test_type == "rfsrc") {
    if (i == 0) {
      rf_nd = rf_nds[which.max(rf_val_perfs)]
      perf = my_predict(test_type, train, test, cols, y_col_train, y_col_test, eq_cases_train_cols = eq_cases_train_cols, rf_nd = rf_nd)
    } else {
      val = test # Need to pass in val to do checks on nodedepth
      perf = my_predict(test_type, train, test, cols,  y_col_train, y_col_test, eq_cases_train_cols = eq_cases_train_cols, val=val)
      rf_nds = c(rf_nds, perf$model$nodedepth)
      rf_val_perfs = c(rf_val_perfs, perf$test$cindex)
    }
  }
  else {
    perf = my_predict(test_type, train, test, cols,  y_col_train, y_col_test, eq_cases_train_cols = eq_cases_train_cols)
  }
  
  if (i == 0) {
    saveRDS(perf, paste0(output, "_perf.rds"))
  }
  
  print(paste(c(i, perf_in_str(test_type, perf)), collapse=", "))
  return(perf_in_1row(test_type, perf))
}

result = as.data.frame(result)
colnames(result) = get_colnames(test_type)
result = rbind(result[nrow(result),],result[1:(nrow(result)-1),]) # Move last row to the top since that's the test case
write(paste(c(test_type, visit_type, num_lag, include_lab, include_ethdon, eq_train_ratio, train_thresh_year, cutoff_visit, # details on train settings
              summarize_perf(test_type, result)), collapse = ","), file = paste0(output, ".txt"))
