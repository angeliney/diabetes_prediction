library("foreach")
library("dplyr")
source('./R/functions.R')
source('./R/features.R')


args = commandArgs(TRUE)
output = as.character(args[1])

df = readRDS("./local_formatted_filtered.rds")

test_types_all_visits = c("mlm", "coxph", "coxphtime", "aft", "glmnetcox", "rfsrc")
test_types_single_visit = c("glm", "glmnet", "coxph", "aft", "glmnetcox", "rfsrc" )
lags = c(0, 3)
booleans = c(T, F)
visit_types = c("all", "first", "random")
train_thresh_years = c(1990, 2000, 2005)

include_lab = T
include_ethdon = F 
cutoff = T

# Get features
features_to_use = features$clin
if (include_lab) features_to_use = c(features_to_use, features$lab)
if (include_ethdon) features_to_use = c(features_to_use, features$eth, features$don)

timedep_cols = intersect(features_to_use, timedep_features)
cov_cols = setdiff(features_to_use, timedep_cols)

# Evaluate
write(paste(c("train_thresh_year", "test_type", "visit_type", "lag",
              "cindex/auroc", "nrow", "cindex_case", "auroc1yr", "auroc3yr", "auroc5yr", 
              sapply(c("min", "median", "mean", "max", "sd", "inffrac"), function(x) paste0(x, "_abs_timediff_case")),
              sapply(c("min", "median", "mean", "max", "sd", "inffrac"), function(x) paste0(x, "_timediff_case")),
              "r2_case", "frac_pred_ge_last_control"), collapse = ","), file = output)

result = foreach(train_thresh_year = train_thresh_years, .combine = rbind) %do% {
  foreach(visit_type = visit_types, .combine = rbind) %do% {
    test_types = {if (visit_type == "all") test_types_all_visits else test_types_single_visit}
    foreach(test_type = test_types, .combine = rbind) %do% {
      foreach(i = lags, .combine = rbind) %do% {
        if (visit_type != "all" & i == 1) return(NULL)
        if (visit_type == "first" && i > 0) return(NULL)
        filename = paste0("output/ethdon", {if (include_ethdon) "T" else "F"}, "_post", train_thresh_year,
                          "_cutoff", {if (cutoff) "T" else "F"},
                          "_", test_type, "_", visit_type, "_lag", i, "_perf.rds")
        if (!file.exists(filename)) {
          return(NULL)
        }
        if (any(grepl(paste(c(train_thresh_year, test_type, visit_type, i), collapse = ","),
                  readLines("local_perf_cutoffT.txt")))) {
          return(NULL)
        }
        print(filename)
        model = readRDS(filename)$model

        # Add lag stuffs
        cols = c(timedep_cols, cov_cols)
        if (i > 0) {
          for (l in 1:i) {
            cols = c(cols, paste0(timedep_cols, "_", l))
          }
        }
        x = as.data.frame(add_lag(df, timedep_cols, i))

        # Get ycol
        if (is_survival(test_type)) {
          if (grepl("time", test_type)) {
            y_col = c("time_since_transplant", "time_next_followup", "diab_in_1_year") # for coxphtime, the is_event status is whether they have diabetes at time of next followup
          } else if (visit_type != "last") {
            y_col = c("time_to_diab", "is_diab")
          } else { # otherwise, for last visit,still use time_to_diab since we want to see if it generalizes
            y_col = c("time_to_diab", "is_diab")
          }
        } else {
          y_col = "diab_in_1_year"
        }

        if (!("time_since_transplant" %in% cols) & !("time_since_transplant" %in% y_col) & !grepl("first", visit_type)) {
          cols = c(cols, "time_since_transplant")
        }

        eval = evaluate_model(x, model, test_type, cols, y_col, include_sroc = T, include_time_diff = T)
        if (is_survival(test_type)) {
          out = cbind(eval$cindex, eval$nrow, eval$cindex_case, eval$auroc1yr, eval$auroc3yr, eval$auroc5yr,
                      summarize_timediff(eval$abs_time_diff_case), summarize_timediff(eval$time_diff_case),
                      eval$r2_case, eval$frac_pred_ge_last_control)
        } else {
          out = cbind(eval$auroc, eval$nrow, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA)
        }
        write.table(cbind(train_thresh_year, test_type, visit_type, i, out), file = output,
                      sep = ",", append = T, row.names = F, quote = F, col.names = F)
        return(cbind(train_thresh_year, test_type, visit_type, i, out))
      }
    }
  }
}