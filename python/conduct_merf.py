import numpy as np
import pandas
import sys
import pickle
sys.path.append('./merf')
from merf import MERF
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from str2bool import str2bool
from functions import prep_exp, filter_train_by_visit, equalize_num_case_control


visit_type = sys.argv[1]
lag = int(sys.argv[2])  # Number of lag variables
include_lab = str2bool(sys.argv[3])  # Include lab features?
include_ethdon = str2bool(sys.argv[4])  # Include ethnicity + donor details?
eq_train_ratio = str2bool(sys.argv[5])  # Train on equal case:control ratio?
output = sys.argv[6]
num_folds = int(sys.argv[7])
train_thresh_year = int(sys.argv[8])  # Where to subset training data
cutoff = str2bool(sys.argv[9])  # Cutoff training data to only include visits pre-2011
if len(sys.argv) < 11:
    file_dir = '/home/diabetes_prediction/'
else:
    file_dir = sys.argv[10]

data = prep_exp(include_lab, include_ethdon, lag, eq_train_ratio, num_folds, train_thresh_year, cutoff,
                file_dir)
cols = np.append(data['cols'], 'time_since_transplant')
nontest_ids = data['train'].drop_duplicates(subset=['TRR_ID', 'is_diab']).TRR_ID

result = pandas.DataFrame()
iters = np.append(range(1, num_folds+1), 0)

max_depths = []
val_aurocs = []

for i in iters:
    if i > 0:
        train_ids = nontest_ids[np.array(data['folds']) != i]
        val_ids = np.setdiff1d(nontest_ids, train_ids)
        test = data['train'][data['train']['TRR_ID'].isin(val_ids)]
    else:
        train_ids = nontest_ids
        test = data['test']

    train = filter_train_by_visit(visit_type, data['train'][data['train'].TRR_ID.isin(train_ids)])
    if eq_train_ratio:
        train = equalize_num_case_control(train, data['eq_cases_train_cols'])

    if visit_type == "all":
        if i > 0:
            val_aurocs2 = []
            for max_depth in [5, 10, 15]:
                model = MERF(n_estimators=100, gll_early_stop_threshold=0.001, max_iterations=2, max_depth=max_depth)
                model.fit(train[cols], pandas.DataFrame(np.ones((train.shape[0], 1))), train.TRR_ID,
                          train.diab_in_1_year)
                test_y_hat = model.predict(test[cols], pandas.DataFrame(np.ones((test.shape[0], 1))), test.TRR_ID)
                test_auroc = roc_auc_score(test.diab_in_1_year, test_y_hat)
                val_aurocs2.append(test_auroc)
            max_depth = [5, 10, 15][np.argmax(val_aurocs2)]
        else:
            max_depth = max_depths[np.argmax(val_aurocs)]

        model = MERF(n_estimators=100, gll_early_stop_threshold=0.001, max_iterations=2, max_depth=max_depth)
        model.fit(train[cols], pandas.DataFrame(np.ones((train.shape[0], 1))), train.TRR_ID, train.diab_in_1_year)
        train_y_hat = model.predict(train[cols], pandas.DataFrame(np.ones((train.shape[0], 1))), train.TRR_ID)
        test_y_hat = model.predict(test[cols], pandas.DataFrame(np.ones((test.shape[0], 1))), test.TRR_ID)
    else:
        if i > 0:
            val_aurocs2 = []
            for max_depth in [5, 10, 15]:
                model = RandomForestClassifier(n_estimators=100, max_depth=max_depth)
                model.fit(train[cols], train.diab_in_1_year)
                test_y_hat = model.predict(test[cols])
                test_auroc = roc_auc_score(test.diab_in_1_year, test_y_hat)
                val_aurocs2.append(test_auroc)
            max_depth = [5, 10, 15][np.argmax(val_aurocs2)]
        else:
            max_depth = max_depths[np.argmax(val_aurocs)]

        model = RandomForestClassifier(n_estimators=100, max_depth=max_depth)
        model.fit(train[cols], train.diab_in_1_year)
        train_y_hat = model.predict(train[cols])
        test_y_hat = model.predict(test[cols])

    train_auroc = roc_auc_score(train.diab_in_1_year, train_y_hat)
    test_auroc = roc_auc_score(test.diab_in_1_year, test_y_hat)
    max_depths.append(max_depth)
    val_aurocs.append(test_auroc)

    if i == 0:
        perf = {'model': model, 'train_auroc': train_auroc, 'test_auroc': test_auroc,
                'train_nrow': train.shape[0], 'test_nrow': test.shape[0]}
        pickle.dump(perf, open('{}_perf.pkl'.format(output), 'wb'))

    print('{},{},{},{},{}'.format(i, train_auroc, test_auroc, train.shape[0], test.shape[0]))
    result = result.append(pandas.DataFrame({'train_auroc': train_auroc, 'test_auroc': test_auroc,
                                             'train_nrow': train.shape[0], 'test_nrow': test.shape[0]}, index=[0]))

result = result.apply(np.roll, shift=1)  # Shift result so that test results is moved from last row to the first row
with open('{}.txt'.format(output), 'w') as f:
    output_lst = ["merf/rf", visit_type, lag, include_lab, include_ethdon, eq_train_ratio, train_thresh_year, cutoff,
                  result["train_nrow"].iloc[0], result["test_nrow"].iloc[0],
                  result["train_auroc"].iloc[0], result["test_auroc"].iloc[0],
                  np.min(result["test_auroc"].iloc[range(1, result.shape[0])]),
                  np.max(result["test_auroc"].iloc[range(1, result.shape[0])])]
    output_str = ",".join(map(str, output_lst))
    f.write(output_str)
