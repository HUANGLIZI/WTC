import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from keras.utils import np_utils

from .NSLKDD_dataset import nsl_kdd
from .AARNet import AARNet


x_train, y_train, x_val, y_val, x_test, y_test = nsl_kdd()

model = AARNet()
# print(model)
label_encoder = LabelEncoder()
criterion = nn.HingeEmbeddingLoss(margin=0.2, size_average=None, reduce=None, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
y_train_econded = label_encoder.transform(y_train)
y_val_econded = label_encoder.transform(y_val)
y_test_econded = label_encoder.transform(y_test)

y_train_dummy = np_utils.to_categorical(y_train_econded)
y_val_dummy = np_utils.to_categorical(y_val_econded)
y_test_dummy = np_utils.to_categorical(y_test_econded)
x_train_values = x_train.values
x_test_values = x_test.values

batch_size = 2000
n_epochs = 50
batch_no = len(x_train) // batch_size

train_loss = 0
train_loss_min = np.Inf
for epoch in range(n_epochs):
    for i in range(batch_no):
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(x_train_values[start:end]))
        y_var = torch.argmax(Variable(torch.LongTensor(y_train_dummy[start:end])), dim=1)
        # y_var2 = torch.argmax(y_var, dim=1)

        optimizer.zero_grad()
        assert isinstance(x_var, object)
        output = model(x_var)

        # output = torch.argmax(outputD)+1
        class_mean_features_norm = x_var / x_var.norm(dim=1)[:, None]
        class_mean_features_norm = torch.abs(class_mean_features_norm)
        # distance_matrix = (1 - torch.mm(class_mean_features_norm, class_mean_features_norm.t()))
        # loss   = class_inter_loss(output,y_var)
        loss = criterion(class_mean_features_norm, -1 * torch.ones_like(class_mean_features_norm).long())
        loss = Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()

        labels = torch.argmax(output, dim=1)
        num_right = np.sum(labels.data.numpy() == y_train_econded[start:end])
        train_loss += loss.item() * batch_size

    train_loss = train_loss / len(x_train)
    print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min, train_loss))
    if train_loss <= train_loss_min:
        print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min, train_loss))
        # print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(y_train[
        # start:end]) ))
        torch.save(model.state_dict(), "model.pt")
        train_loss_min = train_loss

print('Training Ended! ')

x_val = x_val.values
x_val_var = Variable(torch.FloatTensor(x_val), requires_grad=False)
with torch.no_grad():
    val_result = model(x_val_var)
data_frame = pd.DataFrame(val_result.numpy(), index=None)
data_frame.to_csv('val_new.csv')
y_val.to_csv('val_label.csv')
