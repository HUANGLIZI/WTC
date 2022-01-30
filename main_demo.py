import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense, Concatenate, BatchNormalization
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from .dataset.NSLKDD_dataset import nsl_kdd


# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title):
    figsize = (14, 14)
    # y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)


def keras_custom_loss(y_true, y_pred):
    categorical_crossentropy_value = categorical_crossentropy(y_true, y_pred)
    mean_squared_error_value = mean_squared_error(y_true, y_pred)
    custom_loss_value = categorical_crossentropy_value + 0.2 * mean_squared_error_value
    return custom_loss_value


if __name__ == '__main__':  # take nsl-kdd dataset as an example
    x_train, y_train, x_val, y_val, x_test, y_test = nsl_kdd()
    seed_random = 315
    N = 3
    x_train_label, x_train_unlabel, y_train_label, y_train_unlabel = train_test_split(
        x_train, y_train, test_size=0.99, random_state=seed_random)
    # return a tensor
    inputs = Input(shape=(121,))
    output_1 = Dense(256, activation='softplus')(inputs)
    output_2 = Dense(128, activation='relu')(output_1)
    a = BatchNormalization()(output_2)
    output_3 = Dense(64, activation='relu')(output_2)
    z = Concatenate(axis=1)([inputs, output_3])
    output_4 = Dense(32, activation='relu')(z)
    predictions = Dense(11, activation='softmax')(output_4)

    # This section creates a model with an input layer and five full connection layers
    deep_model = Model(inputs=inputs, outputs=predictions)

    deep_model.compile(loss=keras_custom_loss,
                       optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),
                       metrics=['accuracy'])
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(y_train),
                                                y_train)

    y_train_econded = label_encoder.transform(y_train)
    y_train_label_econded = label_encoder.transform(y_train_label)
    y_train_unlabel_econded = label_encoder.transform(y_train_unlabel)
    y_val_econded = label_encoder.transform(y_val)
    y_test_econded = label_encoder.transform(y_test)

    y_train_dummy = np_utils.to_categorical(y_train_econded)
    y_train_label_dummy = np_utils.to_categorical(y_train_label_econded)
    y_train_unlabel_dummy = np_utils.to_categorical(y_train_unlabel_econded)
    y_val_dummy = np_utils.to_categorical(y_val_econded)
    y_test_dummy = np_utils.to_categorical(y_test_econded)

    # start training
    deep_model.fit(x_train_label, y_train_label_dummy,
                   epochs=300,
                   batch_size=3000,
                   validation_data=(x_val, y_val_dummy), class_weight=weights)

    for i in range(N):
        deep_train_pred = deep_model.predict(x_train_unlabel)
        deep_train_pred = np.argmax(deep_train_pred, axis=1)
        deep_train_pred_decoded = label_encoder.inverse_transform(deep_train_pred)
        y_unlabel_econded = label_encoder.transform(deep_train_pred_decoded)
        y_unlabel_dummy = np_utils.to_categorical(y_unlabel_econded)
        deep_model.fit(x_train_unlabel, y_unlabel_dummy,
                       epochs=25,
                       batch_size=2500,
                       validation_data=(x_val, y_val_dummy), class_weight=weights)
        deep_model.fit(x_train_label, y_train_label_dummy,
                       epochs=100,
                       batch_size=2500,
                       validation_data=(x_val, y_val_dummy), class_weight=weights)
    deep_model.fit(x_train_unlabel, y_unlabel_dummy,
                   epochs=25,
                   batch_size=2500,
                   validation_data=(x_val, y_val_dummy), class_weight=weights)

    # deep_model.save("/output/Best_model.hdf5") # save the model

    # start testing
    deep_val_pred = deep_model.predict(x_val)
    deep_val_pred = np.argmax(deep_val_pred, axis=1)
    deep_val_pred_decoded = label_encoder.inverse_transform(deep_val_pred)

    deep_test_pred = deep_model.predict(x_test)
    deep_test_pred = np.argmax(deep_test_pred, axis=1)
    deep_test_pred_decoded = label_encoder.inverse_transform(deep_test_pred)

    plot_cm(y_test, deep_test_pred_decoded, 'Confusion matrix for predictions on the testing set')
    accuracy = accuracy_score(y_test, deep_test_pred_decoded)
    # precision = precision_score(y_val, deep_val_pred_decoded,average='macro')
    # recall = recall_score(y_val, deep_val_pred_decoded, average='binary')
    macro_f1 = f1_score(y_test, deep_test_pred_decoded, average='macro')
    print('accuracy:', accuracy)
    # print('precision:',precision)
    # print('recall:',recall)
    print('Macro-F1: {}'.format(macro_f1))