import numpy as np
import os
import sys
import pandas

sys.path.append('./python/')
from functions import df_to_padded, mod_time
sys.path.append('./theanomodels/')
from utils.misc import readPickle
sys.path.append('./dmm/')
from model_th.dmm import DMM


def get_dmm(unique_id, df, cols, dim_latent):
    dmm = reload_dmm(unique_id)
    df['time2'] = df.time_since_transplant.apply(mod_time)
    df['t_elapsed'] = df.groupby('TRR_ID', group_keys=False).apply(lambda g: g.time2 - g.time2.min())
    df.t_elapsed = df.t_elapsed.astype(int)
    x, id = df_to_padded(df=df, column_names=cols, id_col='TRR_ID', t_col='t_elapsed')
    latent = get_latent_space(dmm, x, id, dim_latent)
    df = df.merge(latent, on=['TRR_ID', 't_elapsed'])
    return df


def reload_dmm(prefix, dmm_dir='./dmm_models/', ep='-EP100'):
    pfile = os.path.join(dmm_dir, prefix + '-config.pkl')
    print 'Hyperparameters in: ', pfile, 'Found: ', os.path.exists(pfile)
    params = readPickle(pfile, quiet=True)[0]

    reload_file = os.path.join(dmm_dir, prefix + ep + '-params.npz')
    print 'Model parameters in: ', reload_file

    # Don't load the training functions for the model since its time consuming
    params['validate_only'] = True
    dmm_reloaded = DMM(params, paramFile=pfile, reloadFile=reload_file)
    return dmm_reloaded


def get_latent_space(dmm, x, id, dim_latent):
    z_q, _, _ = DMM._q_z_x(dmm, x)
    z_q_arr = z_q.eval()
    temp = np.array([np.hstack((z_q_arr[i, :, :],
                    np.ones((z_q_arr.shape[1], 1)) * id[i],
                    np.reshape(range(1, z_q_arr.shape[1]+1), (z_q_arr.shape[1], 1)),
                    )) for i in range(len(id))])
    arr = np.concatenate(temp, axis=0)
    return pandas.DataFrame(data=arr, columns=np.concatenate((['dmm{}'.format(i) for i in range(int(dim_latent))],
                                                             ['TRR_ID', 't_elapsed'])))
