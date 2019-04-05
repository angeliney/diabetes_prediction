# Functions mainly used for processing local data
read_sheets = function(filename, sheets_included = NULL) {
  require("readxl")
  sheetnames = {if (is.null(sheets_included)) excel_sheets(filename) else sheets_included}
  
  sheets = lapply(sheetnames, read_excel, path = filename)
  names(sheets) = sheetnames
  return(sheets)
}

get_year = function (date) {
  return(as.numeric(format(date, "%Y")))
}

check_year_range = function (t, a, b) {
  return(t %in% seq(get_year(a), get_year(b)))
}

# Functions for preprocessing data
to_binary = function(vector, yes_values, no_values, na_values) {
  result = rep(0, length(vector))
  result[!is.na(vector) & vector %in% yes_values] = 1
  result[!is.na(vector) & vector %in% no_values] = 0
  result[is.na(vector) | vector %in% na_values] = NA
  return(result)
}

is_event = function(x) {
  # Figure out who has diabetes at any point in time
  # is_event(c(NA,NA,NA,NA)) == NA
  # is_event(c(NA, 0, 1, NA, 0)) == 1
  # is_event(c(NA, 0, 0, NA)) == 0
  if (all(is.na(x))) NA
  else as.numeric(any(x == 1, na.rm = T))
}

categorize_diagnosis = function(df) {
  df$HEPC = as.numeric(df$DGN_4216 | df$DGN_4593)
  df$HEPB = as.numeric(df$DGN_4592)
  df$NAFLD = as.numeric(df$DGN_4270)
  df$alcohol = as.numeric(df$DGN_4215 | df$DGN_4216 | df$DGN_4217)
  df$PBC = as.numeric(df$DGN_4220)
  df$PSC = as.numeric(df$DGN_4240 | df$DGN_4241 | df$DGN_4242 | df$DGN_4245)
  df$AUTOIMMUNE_HEPATITIS = as.numeric(df$DGN_4212)
  return(df)
}

combine_tx_txf = function(tx, txf, cov_cols, timedep_cols, lags = 0) {
  require("dplyr")
  # Incorporate data from tx_train for those that are constant, e.g. age at time of transplant, gender, race, etc.
  txf = filter(txf, time_since_transplant > 0)
  x = merge(tx[, c(cov_cols, "TRR_ID", "diab_time", "is_diab", "REC_TX_DT", "transplant_year")], txf[, c("TRR_ID", timedep_cols, "time_since_transplant")], by = "TRR_ID")
  
  # Add tx to x putting time_since_followup to 0
  records_to_be_added = cbind(tx[, c("TRR_ID", cov_cols, "diab_time", "is_diab", "REC_TX_DT", "transplant_year")], tx[, timedep_cols], 0)
  colnames(records_to_be_added) = colnames(x)
  
  x = rbind(x, records_to_be_added)
  
  # Add other info that is dependent of time
  x = rename(x, diab_time_since_tx = diab_time)
  x$time_to_diab = x$diab_time_since_tx - x$time_since_transplant
  x$diab_in_1_year = as.numeric(!is.na(x$time_to_diab) & x$time_to_diab <= 1 & !is.na(x$is_diab) & x$is_diab == 1)
  x$diab_now = as.numeric(!is.na(x$time_to_diab) & x$time_to_diab <= 0 & !is.na(x$is_diab) & x$is_diab == 1)
  x$age = round(x$REC_AGE_IN_MONTHS_AT_TX/12 + x$time_since_transplant)
  
  for (col in unique(x$transplant_year[!is.na(x$transplant_year)])) {
    x[, paste0("transplant_year_", col)] = to_binary(x$transplant_year, col, c(), NA)
  }
  
  x = add_lag(x, timedep_cols, lags)
  return(x)
}

replaceNaWithLatest <- function(dfIn, nameColNa = names(dfIn)[1]){
  require("data.table")
  dtTest <- data.table(dfIn)
  setnames(dtTest, nameColNa, "colNa")
  dtTest[, segment := cumsum(!is.na(colNa))]
  dtTest[, colNa := colNa[1], by = "segment"]
  dtTest[, segment := NULL]
  setnames(dtTest, "colNa", nameColNa)
  return(dtTest)
}

add_lag = function(x, timedep_cols, lags) {
  require("dplyr")
  
  x = x %>%
    group_by(TRR_ID) %>%
    arrange(time_since_transplant) %>%
    mutate(time_next_followup = lead(time_since_transplant, default = last(time_since_transplant))) %>%
    replaceNaWithLatest(., "diab_in_1_year") %>%
    replaceNaWithLatest(., "diab_now") %>%
    filter(time_since_transplant <= diab_time_since_tx)
  
  for (timedep_col in timedep_cols) {
    if (!grepl("immuno", timedep_col)) {
      x = x %>%
        group_by(TRR_ID) %>%
        arrange(time_since_transplant) %>%
        replaceNaWithLatest(., timedep_col)
    } # Fill missingness from last observations for timedep cols
    
    if (lags > 0) {
      for (lag_n in 1:lags) {
        x = lag_column = x %>%
          group_by(TRR_ID) %>%
          arrange(time_since_transplant) %>%
          mutate(!!paste0(timedep_col, "_", lag_n) := lag(!! sym(timedep_col), n = lag_n, default = first(!! sym(timedep_col))))
      }
    }
  }
  return(x)
}

# Check data after being merged/combined with combine_tx_txf
visualize_data_checks <- function (data) {
  # Check #1: distribution of features by year of transplant
  ft_by_year = data[complete_rows,] %>% select(cols, "is_diab", "transplant_year") %>% group_by(transplant_year) %>% summarise_all(mean)
  
  # Plot how the distribution change
  normalized = cbind(year=ft_by_year$transplant_year, sweep(ft_by_year[,2:length(colnames(ft_by_year))], 2, colSums(ft_by_year[, 2:length(colnames(ft_by_year))]), FUN="/"))
  normalized.m = melt(normalized, id="year")
  
  ggplot(normalized.m, aes(variable, year))  + geom_tile(aes(fill=value)) + scale_fill_gradient(low = "white", high = "steelblue")  + 
    theme(axis.text.x = element_text(angle = 75, hjust = 1))
  
  # Get the statistics
  stats = ft_by_year %>% select(-year, -REC_TX_DT) %>% summarise_all(.funs = c(mean="mean", min="min", max="max", var="var"))
  
  stats_df = foreach(column = setdiff(colnames(ft_by_year), "transplant_year"), .combine = rbind) %do% {
    cbind(column, as.numeric(stats[paste0(column, "_mean")]), as.numeric(stats[paste0(column, "_min")]), 
          as.numeric(stats[paste0(column, "_max")]), as.numeric(stats[paste0(column, "_var")]))
  }
  
  # Check #2: distribution of features by for followup data number of years after transplant
  complete_rows_train = complete.cases(txf_li_study %>% select(timedep_cols, "time_since_transplant"))
  ft_by_year = txf_li_study[complete_rows,] %>% select(timedep_cols, "time_since_transplant") %>% 
    mutate( year =round(time_since_transplant)) %>% group_by(year) %>% summarise_all(mean)
  
  # Plot how the distribution change
  normalized = cbind(year=ft_by_year$year, sweep(ft_by_year[,2:16], 1, colSums(ft_by_year[, 2:16]), FUN="/"))
  normalized <- cbind(year=ft_by_year$year,data.frame(lapply(ft_by_year[,2:16], function(x) scale(x, center = FALSE, scale = max(x, na.rm = TRUE)/100))))
  normalized.m = melt(normalized, id="year")
  
  ggplot(normalized.m, aes(variable, year))  + geom_tile(aes(fill=value)) + scale_fill_gradient(low = "white", high = "steelblue")  + 
    theme(axis.text.x = element_text(angle = 75, hjust = 1))
}

equalize_num_case_control = function (df, id_col, y_col) {
  require("dplyr")
  case_ids = unique(df %>% filter(!!rlang::sym(y_col) == 1) %>% .[, id_col])
  control_ids = unique(df %>% filter(!!rlang::sym(y_col) == 0) %>% .[, id_col])
  num = min(length(case_ids), length(control_ids))
  
  df2 = filter(df, !!rlang::sym(id_col) %in% c(sample(case_ids, num), sample(control_ids, num)))
  return(df2)
}

my_predict = function(type, train, test, cols, y_col_train, y_col_test, glmnet_alpha = 0, 
                      val = NULL, hidden_dim = 50, python_bin = "/Users/angeliney/miniconda3/bin/python",
                      eq_cases_train_cols = NULL, rf_nd = -1) {
  y_col = y_col_train
  complete_rows_train = complete.cases(train %>% select(cols, y_col))
  num_complete_rows_train = sum(complete_rows_train)
  if (num_complete_rows_train <= 0) {
    print("no complete rows to train on")
    return(list())
  }
  
  train = train[complete_rows_train, ]
  # Take subset of train to get equal numbers of cases and controls
  if (!is.null(eq_cases_train_cols)) {
    train = equalize_num_case_control(train, eq_cases_train_cols[1], eq_cases_train_cols[2])
  }
  
  # Remove columns with only 1 unique value
  for (col in cols) {
    if (length(unique(train[, col])) < 2) cols = setdiff(cols, col)
  }
  
  # Remove duplicated columns
  train = train[,!duplicated(t(train))]
  cols = cols[cols %in% colnames(train)]
  
  # Classification methods
  model = {
    if (type == "rf") {
      require("randomForest")
      x = train %>% select(cols) %>% as.matrix
      y = factor(train[, y_col], levels=c(0, 1))
      
      randomForest(x, y)
  
    } else if (type == "glm") {
      glm(as.formula(paste(y_col, "~", paste(cols, collapse = "+"))), family = binomial, data = train)
      
    } else if (type == "glmnet") {
      require("glmnet")
      xtrain = train %>% select(cols) %>% as.matrix
      ytrain = factor(train[, y_col], levels=c(0, 1))
      
      cv.glmnet(xtrain, ytrain, family = "binomial", alpha = glmnet_alpha)
      
    } else if (type == "mlm") {
      require("lme4")
      glmer(as.formula(paste(y_col, "~", paste(cols, collapse="+"),  "+ (1 | TRR_ID)")), family = binomial, 
            data = train, nAGQ = 0, control = glmerControl(optimizer="nloptwrap"))
      
    } else if (type == "mlmlasso") {
      require("glmmLasso")
      vals = foreach(lambda = 10**seq(-4, -1), .combine = rbind) %do% {
        mdl  = glmmLasso(as.formula(paste(y_col, "~", paste(cols, collapse="+"))), rnd = list(TRR_ID=~1), 
                           family=binomial(link = "logit"), lambda=lambda, data=train)
        val_perf = evaluate_model(mdl, type, val, cols, y_col)
        return(cbind(lambda, val_perf$auroc))
      }
      vals$lambda = as.numeric(vals$lambda)
      vals$roc_val_auc = as.numeric(vals$roc_val_auc)
      lambda = vals$lambda[which.max(vals$roc_val_auc)]
      glmmLasso(as.formula(paste(y_col, "~", paste(cols, collapse="+"))), rnd = list(TRR_ID=~1), 
                family=binomial(link = "logit"), lambda=lambda, data=train)
      
    } 
    
    # Survival methods
    else if (type == "rfsrc") {
      require("randomForestSRC")
      if (!is.null(val)) {
        if (sum(complete.cases(select(val, cols, y_col))) > 0) { 
          # If validation exists, get rf_nd from validation
          vals = foreach(nd = c(5, 10, 15), .combine = rbind) %do% {
            mdl  = rfsrc(as.formula(paste("Surv(", paste(y_col, collapse = " , "), ") ~", 
                                           paste(cols, collapse = " + "))), data =  train, ntree = 100, 
                          nodedepth = nd)
            
            val_perf = evaluate_model(val, mdl, type, cols, y_col_test, include_sroc = F, include_time_diff = F)
            return(cbind(nd, val_perf$cindex))
          }
          nds = as.numeric(vals[, 1])
          roc_val_aucs = as.numeric(vals[, 2])
          rf_nd = nds[which.max(roc_val_aucs)]
        }
      }

      rfsrc(as.formula(paste("Surv(", paste(y_col, collapse = " , "), ") ~", 
                             paste(cols, collapse = " + "))), data =  train, ntree = 100, nodedepth = rf_nd)
      
    } else if (grepl("coxph", type) | (grepl("aft", type))) {
      require("survival")
      survfun = match.fun({ if (grepl("coxph", type)) "coxph" else "survreg" })
      
      survfun(as.formula(paste("Surv(", paste(y_col, collapse = " , "), ") ~", 
                               paste(cols, collapse = " + "), 
                               if (grepl("mixed", type)) "+ frailty(TRR_ID)" else "")), data =  train, x = T)
      
    } else if (type == "glmnetcox") {
      require("survival")
      require("glmnet")
      xtraincox = train %>% select(cols) %>% as.matrix()
      train$time = train[, y_col[1]]
      train$status = train[, y_col[2]]
      ytraincox = train %>% select(time, status)  %>% as.matrix()
      
      cv.glmnet(xtraincox, ytraincox, family = "cox", alpha = glmnet_alpha)
      
    }
  }
  
  train_perf = evaluate_model(train, model, type, cols, y_col, include_sroc = F, include_time_diff = F)
  val_perf = evaluate_model(val, model, type, cols, y_col_test)
  test_perf = evaluate_model(test, model, type, cols, y_col_test)
  
  # Train performance for mixed effect should be different
  if (grepl("mixed", type)) { # mixed effect survival 
    pred = predict(model)
    if (grepl("aft", type)) pred = -1*pred  # aft's lp is the opposite
    train_perf$cindex = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = ", "), ") ~ pred ")), 
                                        train)$concordance
    
  } else if (type %in% c("mlm", "mlmlasso")) { # mixed effect glm
    train_perf$auroc = roc(predictor = predict(model, type = "response"), response = train[, y_col])$auc[1]
  }
  
  output = list(model = model, train = train_perf, val = val_perf, test = test_perf)
  return(output)
}

evaluate_model = function(df, model, type, cols, y_col, include_sroc = T, include_time_diff = T) {
  require("pROC")
  
  results = list()
  num_complete_rows = 0
  if (!is.null(df)) {if (nrow(df) > 0) {
    complete_rows = complete.cases(df %>% select(cols, y_col))
    num_complete_rows = sum(complete_rows)
    results[["nrow"]] = num_complete_rows
    df = df[complete_rows, ]
  }}
  
  if (num_complete_rows <= 0) {
    return(results)
  }
  
  # Classification methods
  if (type == "rf") {
    require("randomForest")
    x = select(df, cols) %>% as.matrix
    y = factor(df[, y_col], levels=c(0, 1))
    results[["auroc"]] = roc(predictor = predict(model, newdata = x, type = "prob")[, 2], response = y)$auc[1]
    
  } else if (type == "glm") {
    results[["auroc"]] = roc(predictor = predict(model, newdata = df, type = "response"), response = df[, y_col])$auc[1]
    
  } else if (type == "glmnet") {
    require("glmnet")
    x = select(df, setdiff(dimnames(coef(model))[[1]], "(Intercept)")) %>% as.matrix
    y = factor(df[, y_col], levels=c(0, 1))
    
    results[["auroc"]] = roc(predictor = predict(model, newx=x, s="lambda.1se", type ="response"), response = y)$auc[1]
    
  } else if (type == "mlm") {
    require("lme4")
    results[["auroc"]] = roc(predictor = predict(model, newdata = df, type = "response", allow.new.levels = T),
                             response = df[, y_col])$auc[1]
    
  } else if (type == "mlmlasso") {
    require("glmmLasso")
    results[["auroc"]] = roc(predictor = predict(model, newdata = df, type = "response"), response = df[, y_col])$auc[1]
    
  } 
  
  # Survival methods
  else if (type == "rfsrc") { 
    require("randomForestSRC")
    require("survival")
    pred = predict(model, df, outcome='train')$predicted
    results[["cindex"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = ", "), ") ~ ", 
                                                           "pred")), df)$concordance
    
    case = filter(df, !!rlang::sym(y_col[2]) == 1)
    pred_case = predict(model, case, outcome='train')$predicted
    results[["cindex_case"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = ", "), ") ~ ", 
                                                           "pred_case")), case)$concordance
    
  } else if (grepl("coxph", type) | grepl("aft", type)) {
    require("survival")
    pred = predict(model, df, type = 'lp')
    if (grepl("aft", type)) pred = -1*pred
    results[["cindex"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = ", "), 
                                                           ") ~ pred")), df)$concordance
    
    case = filter(df, !!rlang::sym(y_col[2]) == 1)
    pred_case = predict(model, case, type = 'lp')
    if (grepl("aft", type)) pred_case = -1*pred_case
    results[["cindex_case"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = ", 
                                                                               "), ") ~ pred_case")), 
                                                                case)$concordance
    
  } else if (type == "glmnetcox") {
    require("survival")
    require("glmnet")
    x = select(df, dimnames(coef(model))[[1]]) %>% as.matrix()
    results[["cindex"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = " , "), ") ~ predict(model, newx=x, type = 'link', s='lambda.1se')")), df)$concordance
    
    case = filter(df, !!rlang::sym(y_col[2]) == 1)
    xcase = select(case, dimnames(coef(model))[[1]]) %>% as.matrix()
    results[["cindex_case"]] = survConcordance(as.formula(paste("Surv(", paste(y_col, collapse = " , "), ") ~ predict(model, newx=xcase, type = 'link', s='lambda.1se')")), case)$concordance
    
  } 
  
  if (is_survival(type)) {
    if (include_time_diff) {
      pred_time = predict_time(df, model, type, cols, y_col)
      
      case_rows = which(df[, y_col[2]] == 1)
      control_rows = which(df[, y_col[2]] == 0)
      results[["abs_time_diff_case"]] = abs(pred_time[case_rows] - df[case_rows, y_col[1]])
      results[["time_diff_case"]] = pred_time[case_rows] -  df[case_rows, y_col[1]]
      
      finite_pred_case = which(is.finite(pred_time[case_rows]))
      case_rows = case_rows[finite_pred_case]
      results[["r2_case"]] = cor(pred_time[case_rows], df[case_rows, y_col[1]],  method = "pearson") 
      
      control = filter(df, !!rlang::sym(y_col[2]) == 0)
      results[["frac_pred_ge_last_control"]] = sum(pred_time[control_rows] >=  df[control_rows, y_col[1]])/length(control_rows)
    
    } else {
      results[["abs_time_diff_case"]] = NA
      results[["time_diff_case"]] = NA
      results[["r2_case"]] = NA
      results[["frac_pred_ge_last_control"]] = NA
    }
    
    if (include_sroc) {
      troc = roc_at_times(df, model, type, cols, y_col, c(1, 3, 5))
      results[["auroc1yr"]] = troc[1]
      results[["auroc3yr"]] = troc[2]
      results[["auroc5yr"]] = troc[3]
    } else {
      for (year in c(1, 3, 5)) {results[[paste0("auroc", year, "yr")]] = NA}
    }
  }
  return(results)
}


predict_time = function(df, model, type, cols, y_col) {
  if (type == "rfsrc") {
    require("randomForestSRC")
    pred = predict(model, df, outcome='train')
    shats = sapply(1:nrow(pred$survival), function(x) {
      value = pred$time.interest[which(pred$survival[x,] < 0.5)[1]]
      return({ if (is.na(value)) Inf else value })
    })
    return(shats)
    
  } else if (grepl("aft", type)) {
    require("survival")
    # Median of the survival function
    return(predict(model, newdata = df, type = "response"))
    
  } else if (grepl("coxph", type)) {
    require("survival")
    # Time when survival just went down under 0.5
    # Since survival function always decreases, just take the first one that is below 0.5
    surv_sum = summary(survfit(model, newdata = df))
    shats = sapply(1:ncol(surv_sum$surv), function(x) {
      value = surv_sum$time[which(surv_sum$surv[, x] < 0.5)[1]]
      return({ if (is.na(value)) Inf else value })
    })
    return(shats)
    
  } else if (type == "glmnetcox") {
    require("glmnet")
    require("hdnom")
    x = select(df, dimnames(coef(model))[[1]]) %>% as.matrix()
    lp = predict(model, newx = x, type = 'link', s='lambda.1se')
    cumhaz = glmnet.basesurv(df[, y_col[1]], df[, y_col[2]], lp, centered = F)
    shats = sapply(lp, function(x) { 
      shat = cumhaz$base_surv ^ exp(x) # no need to demean, since we're using centered = F earlier
      value = cumhaz$times[which(shat < 0.5)[1]]
      return({ if (is.na(value)) Inf else value })
    })
    return(shats)
    
  }
}

roc_at_times = function(df, model, type, cols, y_col, times) {
  require("pROC")
  results = c()
  for (time in times) {
    y = df[, y_col[2]] == 1 & df[, y_col[1]] < time
    surv_prob = predict_surv_at_time(df, model, type, cols, y_col, time)
    event_prob = 1 - surv_prob 
    auroc = roc(predictor = event_prob, response = y)$auc[1]
    results = c(results, auroc)
  }
  return(results)
}

predict_surv_at_time = function(df, model, type, cols, y_col, time) {
  if (type == "rfsrc") {
    require("randomForestSRC")
    require("pec")
    return(predictSurvProb(model, df, time))
    
  } else if (grepl("aft", type)) {
    require("survival")
    # Following CFC::cfc.survreg.survprob
    distname = model$dist
    mydist = unlist(unname(survreg.distributions[distname]))
    cols = names(model$coefficients)[2:length(model$coefficients)]
    if (any(names(mydist) == "dist")) {
      mydist.base = unlist(unname(survreg.distributions[mydist$dist]))
      density = mydist.base$density
      trans  = mydist$trans
    } else {
      density = mydist$density
      trans = function(x) x
    }
    p = sapply(1:nrow(df), function(i) {return(density((trans(time) - t(model$coefficients) %*% 
                                                   t(as.matrix(cbind(1, df[i, cols]))))/model$scale)[, 1])
    })

    return(1 - p)
    
  } else if (grepl("coxph", type)) {
    require("survival")
    require("pec")
    return(predictSurvProb(model, df, time)[, 1])
    
  } else if (type == "glmnetcox") {
    require("glmnet")
    require("hdnom")
    # Slightly modified from hdnom::glmnet.survcurve
    x = select(df, dimnames(coef(model))[[1]]) %>% as.matrix()
    lp = predict(model, newx = x, type = 'link', s='lambda.1se')
    basesurv = glmnet.basesurv(df[, y_col[1]], df[, y_col[2]], lp, time)
    p = exp(exp(lp) %*% (-t(basesurv$cumulative_base_hazard)))
    return(p[,1])
    
  }
  return(NULL)
}

# Functions to output results based on whether it's classification or survival
perf_in_str = function(test_type, perf) {
  if (is_survival(test_type)) {
    return(paste(c(perf$train$cindex, perf$test$cindex, perf$train$nrow, perf$test$nrow,
                  perf$train$cindex_case, perf$test$cindex_case,
                  perf$test$auroc1yr, perf$test$auroc3yr, perf$test$auroc5yr), collapse=", "))
  } else {
    return(paste(c(perf$train$auroc, perf$test$auroc, perf$train$nrow, perf$test$nrow), collapse=", "))
  }
}

perf_in_1row = function(test_type, perf) {
  if (is_survival(test_type)) {
    return(cbind(perf$train$cindex, perf$test$cindex, perf$train$nrow, perf$test$nrow,
                 perf$train$cindex_case, perf$test$cindex_case,
                 perf$test$auroc1yr, perf$test$auroc3yr, perf$test$auroc5yr,
                 summarize_timediff(perf$test$abs_time_diff_case), 
                 summarize_timediff(perf$test$time_diff_case),
                 perf$test$r2_case, perf$test$frac_pred_ge_last_control))
  }
  else {
    return(cbind(perf$train$auroc, perf$test$auroc, perf$train$nrow, perf$test$nrow))
  }
}

get_colnames = function(type) {
  if (is_survival(type)) {
    return(c("train_cindex", "test_cindex", "train_nrow", "test_nrow", "train_cindex_case", "test_cindex_case",
             "test_auroc1yr", "test_auroc3yr", "test_auroc5yr",
             sapply(c("min", "median", "mean", "max", "sd", "inffrac"), function(x) paste0("test_", x, "_abs_timediff_case")),
             sapply(c("min", "median", "mean", "max", "sd", "inffrac"), function(x) paste0("test_", x, "_timediff_case")),
             "test_r2_case", "test_frac_pred_ge_last_control"))
  } else {
    return(c("train_auroc", "test_auroc", "train_nrow", "test_nrow"))
  }
}

summarize_timediff = function(x) {
 return(cbind(t(summary(x[is.finite(x)])[c(1,3,4,6)]), sd(x[is.finite(x)], na.rm = T), sum(is.infinite(x))/length(x)))
}

summarize_perf = function(test_type, output) {
  if (is_survival(test_type)) {
    return(paste(c(output[1, "train_nrow"], output[1, "test_nrow"],
                  output[1, "train_cindex"], output[1, "test_cindex"],
                  min(output[2:nrow(output), "test_cindex"], na.rm = T), 
                  max(output[2:nrow(output), "test_cindex"], na.rm = T), # validation
                  
                  output[1, "train_cindex_case"], output[1, "test_cindex_case"],
                  min(output[2:nrow(output), "test_cindex_case"], na.rm = T), 
                  max(output[2:nrow(output), "test_cindex_case"], na.rm = T), # validation
                  
                  output[1, "test_auroc1yr"],
                  min(output[2:nrow(output), "test_auroc1yr"], na.rm = T), 
                  max(output[2:nrow(output), "test_auroc1yr"], na.rm = T),
                  
                  output[1, "test_auroc3yr"],
                  min(output[2:nrow(output), "test_auroc3yr"], na.rm = T), 
                  max(output[2:nrow(output), "test_auroc3yr"], na.rm = T),
                  
                  output[1, "test_auroc5yr"], 
                  min(output[2:nrow(output), "test_auroc5yr"], na.rm = T), 
                  max(output[2:nrow(output), "test_auroc5yr"], na.rm = T),
                  
                  output[1, "test_r2_case"],
                  min(output[2:nrow(output), "test_r2_case"], na.rm = T), 
                  max(output[2:nrow(output), "test_r2_case"], na.rm = T),
                  
                  output[1, "test_frac_pred_ge_last_control"],
                  min(output[2:nrow(output), "test_frac_pred_ge_last_control"], na.rm = T), 
                  max(output[2:nrow(output), "test_frac_pred_ge_last_control"], na.rm = T),
                  
                  output[1, "test_inffrac_timediff_case"], #inffrac of time difference for cases in test set
                  output[1, "test_min_abs_timediff_case"], output[1, "test_median_abs_timediff_case"], 
                  output[1, "test_mean_abs_timediff_case"], output[1, "test_max_abs_timediff_case"], 
                  output[1, "test_sd_abs_timediff_case"],
                  
                  output[1, "test_min_timediff_case"], output[1, "test_median_timediff_case"], 
                  output[1, "test_mean_timediff_case"], output[1, "test_max_timediff_case"], 
                  output[1, "test_sd_timediff_case"]
                 
    ), collapse = ","))
  } else {
    return(paste(c(output[1, "train_nrow"], output[1, "test_nrow"],
                   output[1, "train_auroc"], output[1, "test_auroc"],
                   min(output[2:nrow(output), "test_auroc"], na.rm = T), 
                   max(output[2:nrow(output), "test_auroc"], na.rm = T)), collapse=",")) # validation
  }
}

is_survival = function(type) { # check if test type is survival
  return (grepl("cox", type) | grepl("aft", type) | type == "rfsrc")
}

get_summary_colnames = function(survival = T) {
  if (survival) {
    return(c("train_nrow", "test_nrow", 
             sapply(c("train", "test", "val_min", "val_max"), function(x) paste0(x, "_cindex")),
             sapply(c("train", "test", "val_min", "val_max"), function(x) paste0(x, "_cindex_case")),
             sapply(c("test", "val_min", "val_max"), function(x) paste0(x, "_auroc1yr")),
             sapply(c("test", "val_min", "val_max"), function(x) paste0(x, "_auroc3yr")),
             sapply(c("test", "val_min", "val_max"), function(x) paste0(x, "_auroc5yr")),
             sapply(c("test", "val_min", "val_max"), function(x) paste0(x, "_r2_case")),
             sapply(c("test", "val_min", "val_max"), function(x) paste0(x, "_frac_pred_ge_last_control")),
             "inffrac_timediff_case",
             sapply(c("min", "median", "mean", "max", "sd"), function(x) paste0("test_", x, "_abs_timediff_case")),
             sapply(c("min", "median", "mean", "max", "sd"), function(x) paste0("test_", x, "_timediff_case"))))
  } else {
    return(c("train_nrow", "test_nrow", 
             sapply(c("train", "test", "val_min", "val_max"), function(x) paste0(x, "_auroc"))))
  }
}