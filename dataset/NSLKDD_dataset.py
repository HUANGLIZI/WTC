import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# copy from the https://www.kaggle.com/nikitaalbert/are-we-under-attack-fe-eda-nn-97-acc

def plot_hist(df, cols, title):
    grid = gridspec.GridSpec(10, 2, wspace=0.5, hspace=0.5)
    fig = plt.figure(figsize=(15, 25))

    for n, col in enumerate(df[cols]):
        ax = plt.subplot(grid[n])

        ax.hist(df[col], bins=20)
        # ax.set_label('Count', fontsize=12)
        ax.set_title(f'{col} distribution', fontsize=15)

    fig.suptitle(title, fontsize=20)
    grid.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()


def nsl_kdd():
    init_train_df = pd.read_csv('../input/nslkdd/kdd_train.csv')
    init_test_df = pd.read_csv('../input/nslkdd/kdd_test.csv')
    random_state = 42
    proc_train_df = init_train_df.copy()  # create a copy of our initial train set to use as our train set.
    proc_test_df = init_test_df.copy()  # create a copy of our initial test set to use as our preprocessed test set.
    proc_train_normal_slice = proc_train_df[
        proc_train_df['labels'] == 'normal'].copy()  # get the slice of our train set with all normal observations
    proc_train_neptune_slice = proc_train_df[
        proc_train_df['labels'] == 'neptune'].copy()  # get the slice of our train set with all neptune observations
    proc_test_normal_slice = proc_test_df[
        proc_test_df['labels'] == 'normal'].copy()  # get the slice of our test set with all normal observations
    proc_test_neptune_slice = proc_test_df[
        proc_test_df['labels'] == 'neptune'].copy()  # get the slice of our test set with all neptune observations
    proc_train_normal_sampled = proc_train_normal_slice.sample(n=5000,
                                                               random_state=random_state)
    # downsample train set normal slice to 5000 observations
    proc_train_neptune_sampled = proc_train_neptune_slice.sample(n=5000,
                                                                 random_state=random_state)
    # downsample train set neptune slice to 5000 observations

    proc_test_normal_sampled = proc_test_normal_slice.sample(n=1000,
                                                             random_state=random_state)
    # downsample test set normal slice to 1000 oberservations
    proc_test_neptune_sampled = proc_test_neptune_slice.sample(n=1000,
                                                               random_state=random_state)
    # downsample test set neptune slice to 5000 oberservations

    proc_train_df.drop(proc_train_df.loc[proc_train_df['labels'] == 'normal'].index,
                       inplace=True)  # drop initial train normal slice
    proc_train_df.drop(proc_train_df.loc[proc_train_df['labels'] == 'neptune'].index,
                       inplace=True)  # drop initial train neptune slice

    proc_test_df.drop(proc_test_df.loc[proc_test_df['labels'] == 'normal'].index,
                      inplace=True)  # drop initial test normal slice
    proc_test_df.drop(proc_test_df.loc[proc_test_df['labels'] == 'neptune'].index,
                      inplace=True)  # drop initial test neptune slice

    proc_train_df = pd.concat([proc_train_df, proc_train_normal_sampled, proc_train_neptune_sampled],
                              axis=0)  # add sampled train normal and neptune slices back to train set
    proc_test_df = pd.concat([proc_test_df, proc_test_normal_sampled, proc_test_neptune_sampled],
                             axis=0)  # add sampled test normal and neptune slices back to test set

    # set(proc_train_df['labels'])
    # set(proc_test_df['labels'])-set(proc_train_df['labels'])
    df2 = proc_train_df[~proc_train_df['labels'].isin(['pod'])]
    df3 = proc_test_df[~proc_test_df['labels'].isin(
        ['pod', 'apache2', 'httptunnel', 'mailbomb', 'snmpgetattack', 'mscan', 'named', 'processtable', 'ps',
         'sendmail',
         'snmpguess', 'xlock', 'xsnoop', 'xterm'])]

    proc_train_df = df2
    proc_test_df = df3

    keep_labels = ['normal', 'neptune', 'satan', 'nmap', 'ipsweep', 'portsweep', 'smurf', 'back', 'teardrop',
                   'warezclient']

    proc_train_df['labels'] = proc_train_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
    proc_test_df['labels'] = proc_test_df['labels'].apply(lambda x: x if x in keep_labels else 'other')

    seed_random = 318
    proc_test_other_slice = proc_test_df[proc_test_df['labels'] == 'other'].copy()
    proc_train_other_sampled, proc_test_other_sampled = train_test_split(proc_test_other_slice, test_size=0.2,
                                                                         random_state=seed_random)
    proc_test_df.drop(proc_test_df.loc[proc_test_df['labels'] == 'other'].index, inplace=True)
    proc_train_df = pd.concat([proc_train_df, proc_train_other_sampled], axis=0)
    proc_test_df = pd.concat([proc_test_df, proc_test_other_sampled], axis=0)
    norm_cols = ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations',
                 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

    for col in norm_cols:
        proc_train_df[col] = np.log(proc_train_df[col] + 1e-6)
        proc_test_df[col] = np.log(proc_test_df[col] + 1e-6)

    proc_train_df['train'] = 1  # add train feature with value 1 to our training set
    proc_test_df['train'] = 0  # add train feature with value 0 to our testing set

    joined_df = pd.concat([proc_train_df, proc_test_df])  # join the two sets

    protocol_dummies = pd.get_dummies(joined_df['protocol_type'],
                                      prefix='protocol_type')  # get one-hot encoded features for protocol_type feature
    service_dummies = pd.get_dummies(joined_df['service'],
                                     prefix='service')  # get one-hot encoded features for service feature
    flag_dummies = pd.get_dummies(joined_df['flag'], prefix='flag')  # get one-hot encoded features for flag feature

    joined_df = pd.concat([joined_df, protocol_dummies, service_dummies, flag_dummies],
                          axis=1)  # join one-hot encoded features to joined dataframe

    proc_train_df = joined_df[joined_df['train'] == 1]  # split train set from joined, using the train feature
    proc_test_df = joined_df[joined_df['train'] == 0]  # split test set from joined, using the train feature

    drop_cols = ['train', 'protocol_type', 'service', 'flag']  # columns to drop

    proc_train_df.drop(drop_cols, axis=1, inplace=True)  # drop original columns from training set
    proc_test_df.drop(drop_cols, axis=1, inplace=True)  # drop original columns from testing set

    # proc_train_df.head() [5, 122]
    # proc_test_df.head() [5, 122]

    y_buffer = proc_train_df['labels'].copy()
    x_buffer = proc_train_df.drop(['labels'], axis=1)

    y_test = proc_test_df['labels'].copy()
    x_test = proc_test_df.drop(['labels'], axis=1)

    seed_random = 315

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_buffer)

    x_train, x_val, y_train, y_val = train_test_split(x_buffer, y_buffer, test_size=0.2, random_state=seed_random)

    return x_train, y_train, x_val, y_val, x_test, y_test
