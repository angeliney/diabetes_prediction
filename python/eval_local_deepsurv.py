import pickle
import pandas
import numpy as np
import os
import deepsurv
import sys
sys.path.append('./python')
from functions import import_local

lags = [0, 3]
train_thresh_years = [1990, 2000, 2005]
visit_types = ["all", "first", "random"]
cutoff = True
f = open('local_perf_deepsurv.txt', 'a+')

for lag in lags:
    local = import_local(lag, '/home/diabetes_prediction/')
    df = local['data']
    for visit_type in visit_types:
        time_col = "time_to_diab"
        cols = data['cols']
        if visit_type != "last":
            cols = np.append(cols, 'time_since_transplant')

        df = df.dropna(subset=cols)
        for train_thresh_year in train_thresh_years:
            filename = "ethdonF_post{}_cutoffT_deepsurv_{}_lag{}_perf.pkl".format(train_thresh_year, visit_type, lag)
            print(filename)
            if os.path.isfile(filename):
                model = pickle.load(open(filename, "rb"))['model']
                y_hat = model.predict(df[cols], pandas.DataFrame(np.ones((df.shape[0], 1))), df.TRR_ID)
                data = {"x": df[cols].values.astype("float32"),
                        "t": df[time_col].values.astype("float32"),
                        "e": df.is_diab.values.astype("int32")}
                cindex = model.get_concordance_index(**data)

                case = {"x": df.query('is_diab == 1')[cols].values.astype("float32"),
                        "t": df.query('is_diab == 1')[time_col].values.astype("float32"),
                        "e": df.query('is_diab == 1').is_diab.values.astype("int32")}
                cindex_case = model.get_concordance_index(**case)
                print('{},deepsurv,{},{},{},{},{}'.format(train_thresh_year, visit_type, lag, cindex, df.shape[0],
                                                          cindex_case))
                f.write('{},deepsurv,{},{},{},{},{}'.format(train_thresh_year, visit_type, lag, cindex, df.shape[0],
                                                            cindex_case))
f.close()
