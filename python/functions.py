import os
import numpy as np


def prep_exp(include_lab, include_ethdon, lag, eq_train_ratio, num_folds, train_thresh_year, cutoff, file_dir):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects.lib.dplyr import DataFrame
    from rpy2.robjects.packages import STAP
    from rpy2.robjects import pandas2ri


    if eq_train_ratio:
        eq_cases_train_cols = np.array(['TRR_ID', 'is_diab'])
    else:
        eq_cases_train_cols = None

    # Read RDS files (load data table)
    read_rds = robjects.r['readRDS']
    tx_li_study = read_rds(os.path.join(file_dir, 'tx_li_formatted.rds'))
    txf_li_study = read_rds(os.path.join(file_dir, 'txf_li_formatted.rds'))

    # Merge them
    cols, cov_cols, timedep_cols = get_cols(include_lab, include_ethdon, lag, file_dir)
    with open(os.path.join(file_dir, 'R', 'functions.R'), 'r') as f:
        string = f.read()
    functions = STAP(string, 'functions')
    merged = functions.combine_tx_txf(tx_li_study, txf_li_study, np.setdiff1d(cov_cols, 'age'), timedep_cols, lag)
    df = pandas2ri.ri2py_dataframe(DataFrame(merged).filter('time_next_followup > time_since_transplant'))

    # Prep data for model training - only take complete ones
    subset_cols = np.concatenate((['TRR_ID', 'age', 'transplant_year'], cols, ['is_diab', 'time_since_transplant',
                                                                               'time_next_followup', 'time_to_diab',
                                                                               'diab_time_since_tx', 'diab_in_1_year',
                                                                               'diab_now']))
    df = df.dropna(subset=subset_cols)
    df_test = df[(df.transplant_year.astype(int) >= 2011) & (df.time_to_diab >= 0)]
    df_nontest = df[(df.transplant_year.astype(int) < 2011) & (df.transplant_year.astype(int) >= train_thresh_year) &
                    (df.time_to_diab >= 0)]
    if cutoff:
        df_nontest = df_nontest[df_nontest.transplant_year.astype(int) + df_nontest.time_since_transplant < 2011]

    if num_folds > 0:
        nontest_y = df_nontest.drop_duplicates(subset=['TRR_ID', 'is_diab']).is_diab
        caret = importr('caret')
        folds = caret.createFolds(nontest_y.values, num_folds, False)
    else:
        folds = None

    return {'test': df_test, 'train': df_nontest, 'cols': cols, 'eq_cases_train_cols': eq_cases_train_cols,
            'folds': folds}


def get_cols(include_lab, include_ethdon, lag, file_dir):
    from rpy2.robjects.packages import STAP
    from rpy2.robjects import numpy2ri

    numpy2ri.activate()

    # Get features based on these inputs
    with open(os.path.join(file_dir, 'R', 'features.R'), 'r') as f:
        string = f.read()
    features_file = STAP(string, 'features')
    features_to_use = features_file.features.rx2('clin')
    if include_lab:
        features_to_use = np.concatenate((features_to_use, features_file.features.rx2('lab')))

    if include_ethdon:
        features_to_use = np.concatenate(
            (features_to_use, features_file.features.rx2('eth'), features_file.features.rx2('don')))

    timedep_cols = np.intersect1d(features_to_use, features_file.timedep_features)
    cov_cols = np.setdiff1d(features_to_use, timedep_cols)

    cols = np.concatenate((timedep_cols, cov_cols))
    if lag > 0:
        for l in range(1, lag + 1):
            cols = np.append(cols, list(map(lambda x: '{}_{}'.format(x, l), timedep_cols)))
    return cols, cov_cols, timedep_cols


def filter_train_by_visit(visit_type, df):
    if visit_type == 'first':
        train = df[((df.transplant_year.astype(int) < 2000) & (df.time_since_transplant == 0.5)) |
                   ((df.transplant_year.astype(int) >= 2000) & (df.time_since_transplant == 0))]
    elif visit_type == "last":
        max_idx = df.groupby("TRR_ID").time_since_transplant.transform(max) == df.time_since_transplant
        train = df[max_idx]
    elif visit_type == "random":
        train = df.groupby("TRR_ID").apply(lambda x: x.sample(n=1))
    else:
        train = df
    return train


def equalize_num_case_control(df, eq_cases_train_cols):
    case_ids = df[df[eq_cases_train_cols[1]] == 1][eq_cases_train_cols[0]].unique()
    control_ids = df[df[eq_cases_train_cols[1]] == 0][eq_cases_train_cols[0]].unique()
    num = min(case_ids.shape[0], control_ids.shape[0])

    include_ids = np.concatenate((np.random.choice(case_ids, num), np.random.choice(control_ids, num)))
    df2 = df[df[eq_cases_train_cols[0]].isin(include_ids)]
    return df2


def import_local(lag, file_dir):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import STAP
    from rpy2.robjects import pandas2ri

    cols, cov_cols, timedep_cols = get_cols(True, False, lag, file_dir)

    read_rds = robjects.r['readRDS']
    df = read_rds(os.path.join(file_dir, 'local_formatted_filtered.rds'))

    with open(os.path.join(file_dir, 'R', 'functions.R'), 'r') as f:
        string = f.read()
    functions = STAP(string, 'functions')

    df = functions.add_lag(df, timedep_cols, lag)
    return {'data': pandas2ri.ri2py_dataframe(df), 'cols': cols}


# Modify time_since_transplant to accommodate for 0.5 value
def mod_time(x):
    from math import ceil
    if x == 0:
        return -1
    elif x == 0.5:
        return 0
    elif x % 1 > 0:
        return ceil(x)  # To prevent collapse with visit time from beforehand, always round up
    else:
        return x


# These are from wtte.transforms but changed, because there's a bug in their versions: id was not ordered
# and the seq_lengths in df_to_array are not in the same order as unique_ids
def df_to_padded(df, column_names, id_col='id', t_col='t', max_seq_len=None):
    """Pads pandas df to a numpy array of shape `[n_seqs,max_seqlen,n_features]`.
        see `df_to_array` for details
    """
    return df_to_array(df, column_names, nanpad_right=True, id_col=id_col, t_col=t_col, max_seq_len=max_seq_len)


def df_to_array(df, column_names, nanpad_right=True, id_col='id', t_col='t', max_seq_len=None):
    """Converts flat pandas df with cols `id,t,col1,col2,..` to array indexed `[id,t,col]`.
    :param df: dataframe with columns:
      * `id`: Any type. A unique key for the sequence.
      * `t`: integer. If `t` is a non-contiguous int vec per id then steps in
        between t's are padded with zeros.
      * `columns` in `column_names` (String list)
    :type df: Pandas dataframe
    :param Boolean nanpad_right: If `True`, sequences are `np.nan`-padded to `max_seq_len`
    :param return_lists: Put every tensor in its own subarray
    :param id_col: string column name for `id`
    :param t_col: string column name for `t`
    :return padded: With seqlen the max value of `t` per id
      a numpy float array of dimension `[n_seqs,max_seqlen,n_features]`
    """

    # Do not sort. Create a view.
    grouped = df.groupby(id_col, sort=False)

    unique_ids = list(grouped.groups.keys())

    n_seqs = grouped.ngroups
    n_features = len(column_names)
    seq_lengths = df[[id_col, t_col]].groupby(id_col, as_index=False).aggregate('max')
    seq_lengths[t_col] = seq_lengths[t_col] + 1

    # We can't assume to fit varying length seqs. in flat array without
    # padding.
    assert nanpad_right or seq_lengths.shape[0] == 1, 'Wont fit in flat array'

    if not max_seq_len:
        max_seq_len = seq_lengths[t_col].values.max()

    # Initialize the array to be filled
    padded = np.zeros([n_seqs, max_seq_len, n_features])

    # Fill it
    for s in xrange(n_seqs):
        # df_user is a view
        df_group = grouped.get_group(unique_ids[s]).query("{} <= {}".format(t_col, max_seq_len))

        padded[s][df_group[t_col].values, :] = df_group[column_names].values

        seq_len = seq_lengths[seq_lengths[id_col] == unique_ids[s]][t_col].values[0]
        if nanpad_right and seq_len < max_seq_len:
            padded[s][seq_len:, :].fill(np.nan)

    return padded, unique_ids
