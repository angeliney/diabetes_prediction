import pandas
import numpy as np
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functions import mod_time, df_to_padded, prep_exp

sys.path.append('./theanomodels/')
from utils.misc import loadHDF5

sys.path.append('./dmm/')
from model_th.dmm import DMM
import model_th.learning as DMM_learn


def train_dmm(train_data, test_data, cols, dim_latent, unique_id, time_col='time_since_transplant', id_col='TRR_ID'):
    df = pandas.concat([train_data, test_data])

    # Format time so that it can be indexed
    df['time2'] = df[time_col].apply(mod_time)

    # This is different than time since transplant since some people miss albumin, bilirubin,
    # creatinine or acute rej episode at time of transplant
    df['t_elapsed'] = df.groupby(id_col, group_keys=False).apply(lambda g: g.time2 - g.time2.min())
    df.t_elapsed = df.t_elapsed.astype(int)

    nontest_ids = train_data[id_col].drop_duplicates()
    train_ids = nontest_ids.sample(frac=0.9)
    train = df[df[id_col].isin(train_ids)]
    val = df[np.logical_not(df[id_col].isin(train_ids))]
    test = df[df[id_col].isin(test_data[id_col])]

    # Reformat to a matrix
    x_train, id_train = df_to_padded(df=train, column_names=cols, id_col=id_col, t_col='t_elapsed')
    x_val, id_val = df_to_padded(df=val, column_names=cols, id_col=id_col, t_col='t_elapsed',
                                 max_seq_len=x_train.shape[1])
    x_test, id_test = df_to_padded(df=test, column_names=cols, id_col=id_col, t_col='t_elapsed',
                                   max_seq_len=x_train.shape[1])
    mask_value = -1.3371337
    x_train_masked = x_train.copy()
    x_train_masked[np.isnan(x_train_masked)] = mask_value
    x_val_masked = x_val.copy()
    x_val_masked[np.isnan(x_val_masked)] = mask_value
    x_test_masked = x_test.copy()
    x_test_masked[np.isnan(x_test_masked)] = mask_value

    dataset = {
        'dim_observations': cols.shape[0],
        'data_type': 'real',
        'train': {'tensor': x_train_masked, 'mask': np.logical_not(np.isnan(x_train[:, :, 0])), 'id': id_train},
        'valid': {'tensor': x_val_masked, 'mask': np.logical_not(np.isnan(x_val[:, :, 0])), 'id': id_val},
        'test': {'tensor': x_test_masked, 'mask': np.logical_not(np.isnan(x_test[:, :, 0])), 'id': id_test},
    }

    max_visits = x_train.shape[1]
    params = {
        'dim_observations': dataset['dim_observations'],
        'data_type': dataset['data_type'],
        'dataset': 'srtr',
        'epochs': 10,
        'seed': 1,
        'init_weight': 0.1,
        'dim_stochastic': dim_latent,
        'expt_name': 'something',
        'reg_value': 0.05,
        'reloadFile': './NOSUCHFILE',
        'reg_spec': '_',
        'dim_hidden': max_visits,
        'lr': 0.0008,
        'reg_type': 'l2',
        'init_scheme': 'uniform',
        'optimizer': 'adam',
        'use_generative_prior': 'approx',
        'maxout_stride': 4,
        'batch_size': 512,
        'savedir': './dmm_models',
        'forget_bias': -5.0,
        'inference_model': 'R',
        'emission_layers': 2,
        'savefreq': 100,
        'rnn_cell': 'lstm',
        'rnn_size': max_visits,
        'paramFile': './NOSUCHFILE',
        'nonlinearity': 'relu',
        'rnn_dropout': 0.1,
        'transition_layers': 2,
        'anneal_rate': 2.0,
        'debug': False,
        'validate_only': False,
        'transition_type': 'mlp',
        'unique_id': unique_id,
        'leaky_param': 0.0
    }

    # Create a temporary directory to save checkpoints
    os.system('mkdir -p ' + params['savedir'])

    # Specify the file where `params` corresponding for this choice of model and data will be saved
    pfile = params['savedir'] + '/' + params['unique_id'] + '-config.pkl'

    print 'Checkpoint prefix: ', pfile
    dmm = DMM(params, paramFile=pfile)

    savef = os.path.join(params['savedir'], params['unique_id'])
    savedata = DMM_learn.learn(dmm, dataset['train'], epoch_start=0,
                               epoch_end=101,
                               batch_size=params['batch_size'],
                               savefreq=params['savefreq'],
                               savefile=savef,
                               dataset_eval=dataset['valid'],
                               shuffle=True)
    return savedata


def plot_loss(unique_id):
    stats = loadHDF5('./dmm_models/{}-EP100-stats.h5'.format(unique_id))
    plt.plot(stats['train_bound'][:, 0], stats['train_bound'][:, 1], '-o', color='g', label='Train')
    plt.plot(stats['valid_bound'][:, 0], stats['valid_bound'][:, 1], '-*', color='b', label='Validate')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Upper Bound on $-\log p(x)$')
    plt.savefig('./dmm_models/{}-EP100-loss.png'.format(unique_id), format='png')


train_thresh_years = [2005, 2000, 1990]
include_lab = True
lag = 0
eq_train_ratio = False
num_folds = 0

for cutoff in [True, False]:
    for include_ethdon in [True, False]:
        for train_thresh_year in train_thresh_years:
            for dim_latent in [2, 8]:
                data = prep_exp(include_lab, include_ethdon, lag, eq_train_ratio, num_folds, train_thresh_year, cutoff,
                                '/h/angeliney/projects/SRTR')
                savedir = "ethdon{}_post{}_cutoff{}_dim{}".format("T" if include_ethdon else "F", train_thresh_year,
                                                                  "T" if cutoff else "F", dim_latent)
                train_dmm(data['train'], data['test'], data['cols'], dim_latent, savedir)
                plot_loss(savedir)
                print("Done {}".format(savedir))
