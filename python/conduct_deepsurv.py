import numpy as np
import pandas
import sys
import pickle
import lasagne
sys.path.append("./deepsurv")
import deepsurv
import optunity
from str2bool import str2bool
from functions import prep_exp, filter_train_by_visit, equalize_num_case_control


visit_type = sys.argv[1]
lag = int(sys.argv[2])  # Number of lag variables
include_lab = str2bool(sys.argv[3])  # Include lab features?
include_ethdon = str2bool(sys.argv[4])  # Include ethnicity + donor details?
eq_train_ratio = str2bool(sys.argv[5])  # Train on equal case:control ratio?
output = sys.argv[6]
num_folds = int(sys.argv[7])
train_thresh_year = int(sys.argv[8])  # Whether to subset training data to data post2000
cutoff = str2bool(sys.argv[9])  # Cutoff training data to only include visits pre-2011
if len(sys.argv) < 11:
    file_dir = '/home/diabetes_prediction/'
else:
    file_dir = sys.argv[10]

data = prep_exp(include_lab, include_ethdon, lag, eq_train_ratio, num_folds, train_thresh_year, cutoff,
                file_dir)
nontest_ids = data['train'].drop_duplicates(subset=['TRR_ID', 'is_diab']).TRR_ID

if visit_type == "last":
    time_col_train = "time_since_transplant"
    time_col_test = "time_to_diab"
    cols = data['cols']
else:
    time_col_train = "time_to_diab"
    time_col_test = time_col_train
    cols = np.append(data['cols'], 'time_since_transplant')


# Find best hyperparameter setting
def get_objective_function(num_epochs, update_fn=lasagne.updates.momentum):
    """
    Returns the function for Optunity to optimize. The function returned by get_objective_function
    takes the parameters: x_train, y_train, x_test, and y_test, and any additional kwargs to
    use as hyper-parameters.
    The objective function runs a DeepSurv model on the training data and evaluates it against the
    test set for validation. The result of the function call is the validation concordance index
    (which Optunity tries to optimize)
    """

    def format_to_deepsurv(x, y):
        return {
            'x': x.astype(np.float32),
            'e': y[:, 0].astype(np.int32),
            't': y[:, 1].astype(np.float32)
        }

    def get_hyperparams(params):
        hyperparams = {
            'batch_norm': True,
            'standardize': True
        }
        if 'num_layers' in params and 'num_nodes' in params:
            params['hidden_layers_sizes'] = [int(params['num_nodes'])] * int(params['num_layers'])
            del params['num_layers']
            del params['num_nodes']
        if 'learning_rate' in params:
            params['learning_rate'] = 10 ** params['learning_rate']
        hyperparams.update(params)
        return hyperparams

    def train_deepsurv(x_train, y_train, x_test, y_test, **kwargs):
        # Standardize the datasets
        train_mean = x_train.mean(axis=0)
        train_std = x_train.std(axis=0)
        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std

        train_data = format_to_deepsurv(x_train, y_train)
        valid_data = format_to_deepsurv(x_test, y_test)

        hyperparams = get_hyperparams(kwargs)

        network = deepsurv.DeepSurv(n_in=train_data['x'].shape[1], **hyperparams)
        metrics = network.train(train_data, n_epochs=num_epochs,
                                update_fn=update_fn, verbose=False)

        result = network.get_concordance_index(**valid_data)
        return result

    return train_deepsurv


update_fn = lasagne.updates.momentum
box_constraints = {
    "learning_rate": [-7, -3],
    "num_nodes": [2, 20],
    "num_layers": [1, 4],
    "lr_decay": [0.0, 0.001],
    "momentum": [0.8, 0.95],
    "L1_reg": [0.05, 5.0],
    "L2_reg": [0.05, 5.0],
    "dropout": [0.0, 0.5]
}
opt_fxn = get_objective_function(100, update_fn=update_fn)
train = filter_train_by_visit(visit_type, data['train'])

opt_fxn = optunity.cross_validated(x=train[cols].values,
                                   y=np.column_stack((train.is_diab.values, train[time_col_train].values)),
                                   num_folds=num_folds)(opt_fxn)
opt_params, call_log, _ = optunity.maximize(opt_fxn, num_evals=50, solver_name='sobol', **box_constraints)
hyperparams = opt_params

hyperparams['hidden_layers_sizes'] = [int(hyperparams['num_nodes'])] * int(hyperparams['num_layers'])
del hyperparams['num_layers']
del hyperparams['num_nodes']
hyperparams['batch_norm'] = True
hyperparams['standardize'] = True
hyperparams['learning_rate'] = 10**hyperparams['learning_rate']

# Start actual model training with best hyper params
result = pandas.DataFrame()
for i in range(num_folds+1):
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

    train_data = {"x": train[cols].values.astype("float32"),
                  "t": train[time_col_train].values.astype("float32"),
                  "e": train.is_diab.values.astype("int32")}

    test_data = {"x": test[cols].values.astype("float32"),
                 "t": test[time_col_test].values.astype("float32"),
                 "e": test.is_diab.values.astype("int32")}

    network = deepsurv.DeepSurv(n_in=train_data['x'].shape[1], **hyperparams)
    log = network.train(train_data, n_epochs=1000, update_fn=update_fn)
    train_cindex = network.get_concordance_index(**train_data)
    test_cindex = network.get_concordance_index(**test_data)

    # Get c-index case
    train_case_data = {"x": train.query('is_diab == 1')[cols].values.astype("float32"),
                       "t": train.query('is_diab == 1')[time_col_train].values.astype("float32"),
                       "e": train.query('is_diab == 1').is_diab.values.astype("int32")}
    test_case_data = {"x": test.query('is_diab == 1')[cols].values.astype("float32"),
                      "t": test.query('is_diab == 1')[time_col_test].values.astype("float32"),
                      "e": test.query('is_diab == 1').is_diab.values.astype("int32")}
    train_cindex_case = network.get_concordance_index(**train_case_data)
    test_cindex_case = network.get_concordance_index(**test_case_data)

    if i == 0:
        perf = {'model': network, 'train_cindex': train_cindex, 'test_cindex': test_cindex,
                'train_nrow': train.shape[0], 'test_nrow': test.shape[0],
                'train_cindex_case': train_cindex_case, 'test_cindex_case': test_cindex_case}
        pickle.dump(perf, open('{}_perf.pkl'.format(output), 'wb'))

    print('{},{},{},{},{},{},{}'.format(i, train_cindex, test_cindex, train.shape[0], test.shape[0], train_cindex_case,
                                        test_cindex_case))
    result = result.append(pandas.DataFrame({'train_cindex': train_cindex, 'test_cindex': test_cindex,
                                             'train_nrow': train.shape[0], 'test_nrow': test.shape[0],
                                             'train_cindex_case': train_cindex_case,
                                             'test_cindex_case': test_cindex_case}, index=[0]))

with open('{}.txt'.format(output), 'w') as f:
    output_lst = ["deepsurv", visit_type, lag, include_lab, include_ethdon, eq_train_ratio, train_thresh_year, cutoff,
                  result["train_nrow"].iloc[0], result["test_nrow"].iloc[0],
                  result["train_cindex"].iloc[0], result["test_cindex"].iloc[0],
                  np.min(result["test_cindex"].iloc[range(1, result.shape[0])]),
                  np.max(result["test_cindex"].iloc[range(1, result.shape[0])]),
                  result["train_cindex_case"].iloc[0], result["test_cindex_case"].iloc[0],
                  np.min(result["test_cindex_case"].iloc[range(1, result.shape[0])]),
                  np.max(result["test_cindex_case"].iloc[range(1, result.shape[0])])]
    output_str = ",".join(map(str, output_lst))
    f.write(output_str)
