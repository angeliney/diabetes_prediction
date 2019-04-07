import pickle
import pandas
import numpy as np
import os
import sys
sys.path.append('./merf')
from merf import MERF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
sys.path.append('./python')
from functions import import_local

lags = [0, 3]
train_thresh_years = [1990, 2000, 2005]
visit_types = ["all", "first", "random"]
cutoff = True

f = open('local_perf_merf.txt', 'a+')
for visit_type in visit_types:
    for lag in lags:
        local = import_local(lag, '/home/diabetes_prediction/')
        df = local['data']
        cols = np.append(local['cols'], "time_since_transplant")
        df = df.dropna(subset=cols)
        for train_thresh_year in train_thresh_years:
            filename = "ethdonF_post{}_cutoffT_merf_{}_lag{}_perf.pkl".format(train_thresh_year, visit_type, lag)
            print(filename)
            if os.path.isfile(os.path.join("output", filename)):
                model = pickle.load(open(os.path.join("output", filename), "rb"))['model']
                if visit_type == "all":
                    y_hat = model.predict(df[cols], pandas.DataFrame(np.ones((df.shape[0], 1))), df.TRR_ID)
                else:
                    y_hat = model.predict(df[cols])

                auroc = roc_auc_score(df.diab_in_1_year, y_hat)
                print('{},merf,{},{},{},{}'.format(train_thresh_year, visit_type, lag, auroc, df.shape[0]))
                f.write('{},merf,{},{},{},{}\n'.format(train_thresh_year, visit_type, lag, auroc, df.shape[0]))
f.close()
