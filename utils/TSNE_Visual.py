import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

init_train_df = pd.read_csv('kdd_train.csv')
init_test_df = pd.read_csv('kdd_test.csv')
random_state = 42
 
proc_train_df = init_train_df.copy()                                                                      # create a copy of our initial train set to use as our preproccessed train set.
proc_test_df = init_test_df.copy()                                                                        # create a copy of our initial test set to use as our preproccessed test set.

proc_train_normal_slice = proc_train_df[proc_train_df['labels']=='normal'].copy()                         # get the slice of our train set with all normal observations
proc_train_neptune_slice = proc_train_df[proc_train_df['labels']=='neptune'].copy()                       # get the slice of our train set with all neptune observations

proc_test_normal_slice = proc_test_df[proc_test_df['labels']=='normal'].copy()                            # get the slice of our test set with all normal observations
proc_test_neptune_slice = proc_test_df[proc_test_df['labels']=='neptune'].copy()                          # get the slice of our test set with all neptune observations

proc_train_normal_sampled = proc_train_normal_slice.sample(n=5000, random_state=random_state)             # downsample train set normal slice to 5000 oberservations
proc_train_neptune_sampled = proc_train_neptune_slice.sample(n=5000, random_state=random_state)           # downsample train set neptune slice to 5000 oberservations

proc_test_normal_sampled = proc_test_normal_slice.sample(n=1000, random_state=random_state)               # downsample test set normal slice to 1000 oberservations
proc_test_neptune_sampled = proc_test_neptune_slice.sample(n=1000, random_state=random_state)             # downsample test set neptune slice to 5000 oberservations

proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='normal'].index, inplace=True)              # drop initial train normal slice
proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='neptune'].index, inplace=True)             # drop initial train neptune slice

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='normal'].index, inplace=True)                 # drop initial test normal slice
proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='neptune'].index, inplace=True)                # drop initial test neptune slice

proc_train_df = pd.concat([proc_train_df, proc_train_normal_sampled, proc_train_neptune_sampled], axis=0) # add sampled train normal and neptune slices back to train set
proc_test_df = pd.concat([proc_test_df, proc_test_normal_sampled, proc_test_neptune_sampled], axis=0)     # add sampled test normal and neptune slices back to test set
keep_labels = ['normal', 'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop', 'warezclient']

proc_train_df['labels'] = proc_train_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
proc_test_df['labels'] = proc_test_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
from sklearn.model_selection import train_test_split

seed_random = 718

proc_test_other_slice = proc_test_df[proc_test_df['labels']=='other'].copy()

proc_train_other_sampled, proc_test_other_sampled = train_test_split(proc_test_other_slice, test_size=0.2, random_state=seed_random)

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='other'].index, inplace=True)

proc_train_df = pd.concat([proc_train_df, proc_train_other_sampled], axis=0)
proc_test_df = pd.concat([proc_test_df, proc_test_other_sampled], axis=0)
norm_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

for col in norm_cols:
    proc_train_df[col] = np.log(proc_train_df[col]+1e-6)
    proc_test_df[col] = np.log(proc_test_df[col]+1e-6)
proc_train_df['train']=1                                                                       # add train feature with value 1 to our training set
proc_test_df['train']=0                                                                        # add train feature with value 0 to our testing set

joined_df = pd.concat([proc_train_df, proc_test_df])                                           # join the two sets
 
protocol_dummies = pd.get_dummies(joined_df['protocol_type'], prefix='protocol_type')          # get one-hot encoded features for protocol_type feature
service_dummies = pd.get_dummies(joined_df['service'], prefix='service')                       # get one-hot encoded features for service feature
flag_dummies = pd.get_dummies(joined_df['flag'], prefix='flag')                                # get one-hot encoded features for flag feature

joined_df = pd.concat([joined_df, protocol_dummies, service_dummies, flag_dummies], axis=1)    # join one-hot encoded features to joined dataframe

proc_train_df = joined_df[joined_df['train']==1]                                               # split train set from joined, using the train feature
proc_test_df = joined_df[joined_df['train']==0]                                                # split test set from joined, using the train feature

drop_cols = ['train', 'protocol_type', 'service', 'flag']                                      # columns to drop

proc_train_df.drop(drop_cols, axis=1, inplace=True)                                            # drop original columns from training set
proc_test_df.drop(drop_cols, axis=1, inplace=True)  
y_buffer = proc_train_df['labels'].copy()
x_buffer = proc_train_df.drop(['labels'], axis=1)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

seed_random = 315

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_buffer)
y_test = proc_test_df['labels'].copy()
x_test = proc_test_df.drop(['labels'], axis=1)
y_buffer_econded = label_encoder.transform(y_buffer)
y_test_econded = label_encoder.transform(y_test)
#x_test=x_buffer.values
#y_test=y_buffer
##T-SNE Visualization.

from sklearn.manifold import TSNE

tsne = TSNE()

tsne1 = TSNE(n_components = 2, random_state =0)

x_test_2d = tsne1.fit_transform(x_test)

##SCTTER PLOTTING THE SAMPLE POINTS
import matplotlib.pyplot as mlt
color_map = {0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",}
mlt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    mlt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], label=cl)
mlt.xlabel('X in t-SNE')
mlt.ylabel('Y in t-SNE')
#box = mlt.get_position()
#mlt.set_position([box.x0, box.y0, box.width , box.height* 0.8])
#mlt.legend(loc='upper left', bbox_to_anchor=(0.5, 1.2),ncol=3)
#mlt.legend(loc='upper left')
mlt.title('t-SNE visualization of test data')
mlt.show()