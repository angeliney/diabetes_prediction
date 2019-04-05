library(dplyr)
library(data.table)

source('/home/diabetes_prediction/functions.R')
source('/home/diabetes_prediction/features.R') # Get features list

args = commandArgs(TRUE)
include_lab = as.logical(args[1]) # Include lab measurement?
include_ethdon = as.logical(args[2]) # Include patient's ethnicity and race, include details of donor info?
lag = as.numeric(args[3]) # Number of lag columns to include, if lag > 10, then it refers to DMM hidden dimension: 12: 2-dim, 13: 3-dim
eq_train_ratio = as.logical(args[4]) # Train on 50:50 ratio
train_thresh_year = as.numeric(args[5]) # Only use data transplant_year >= train_thresh_year 
cutoff_visit = as.logical(args[6]) # For training data, only use visits before 2011

# Get features based on input
features_to_use = features$clin
if (include_lab) features_to_use = c(features_to_use, features$lab)
if (include_ethdon) features_to_use = c(features_to_use, features$eth, features$don)

timedep_cols = intersect(features_to_use, timedep_features)
cov_cols = setdiff(features_to_use, timedep_cols)

eq_cases_train_cols = if (eq_train_ratio) c("TRR_ID", "is_diab") else NULL

# Merge
tx_li_study = readRDS("tx_li_formatted.rds")
txf_li_study = readRDS("txf_li_formatted.rds")
# age is generated within combine_tx_tf so remove from columns
if (lag <= 10) {
  df = combine_tx_txf(tx_li_study, txf_li_study, setdiff(cov_cols, "age"), timedep_cols, lags = lag) %>% 
    filter(time_next_followup > time_since_transplant) %>% as.data.frame
} else {
  df = combine_tx_txf(tx_li_study, txf_li_study, setdiff(cov_cols, "age"), timedep_cols) %>% 
    filter(time_next_followup > time_since_transplant) %>% as.data.frame
  # Get DMM
  require("reticulate")
  source_python("python/get_dmm.py")
  df = get_dmm(paste0("ethdon", {if (include_ethdon) "T" else "F"}, 
                                  "_post", train_thresh_year, 
                                  "_cutoff", {if (include_ethdon) "T" else "F"},
                                  "_dim", lag-10), df, features_to_use, lag-10)
}

# Prep data for model training
cols = c(timedep_cols, cov_cols)
if (lag > 0 & lag <= 10) {
  if (lag <= 10) {
    for (l in 1:lag) {
      cols = c(cols, paste0(timedep_cols, "_", l))
    }
  } else {
    # DMM
    for (l in 11:lag) {
      cols = c(cols, paste0("dmm", l-10))
    }
  }
}
subset_cols = c("TRR_ID", "age", cols, "is_diab", "time_since_transplant", "time_next_followup", "time_to_diab", "diab_time_since_tx", "diab_in_1_year", "diab_now", "transplant_year", colnames(df)[grepl("transplant_year_", colnames(df))])
complete_rows = complete.cases(df %>% select(subset_cols))
df = df[complete_rows, subset_cols]

df_test = filter(df, as.numeric(transplant_year) >= 2011, time_to_diab >= 0, !is.na(is_diab))
df_nontest = filter(df, as.numeric(transplant_year) < 2011, as.numeric(transplant_year) >= train_thresh_year,
                    time_to_diab >= 0, !is.na(is_diab))

if (cutoff_visit) {
  df_nontest = filter(df_nontest, as.numeric(transplant_year) + time_since_transplant < 2011)
}
