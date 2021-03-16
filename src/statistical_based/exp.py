from feature import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import matplotlib as mp
mp.use('agg')


def read_data():
    data = np.load('data/dataset.npy')
    label = np.load('data/label.npy')
    index = np.logical_and(label != 12, label != 14)  # fix label
    data, label = data[index], label[index]
    return data, label


def feature_extractor(x):

    features = basic_info(x) + startend(x) + absolute_sum_of_changes(x) + distribution(x) + count_with_mean(x) + location(x) + cid_ce(x)
    #features = features + transform_min(x)

    return features


data, label = read_data()
# feature extraction

train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.25, random_state=40)

train_feature, test_feature = [], []
for i in train_data:
    train_feature.append(feature_extractor(i))
for i in test_data:
    test_feature.append(feature_extractor(i))
#print(np.array(train_data).shape, np.array(train_label).shape)

model = RandomForestClassifier(n_estimators=100, random_state=10)
model.fit(train_feature, train_label)

model.fit(train_feature, train_label)

pred_y = model.predict(test_feature)
p, r, f, _ = precision_recall_fscore_support(
    test_label, pred_y, labels=range(13), average="weighted")
print(model.feature_importances_)
print(f"precision: {p}\nrecall: {r}\nf-score: {f}")

false_index = []
for i in range(len(test_data)):
    if test_label[i] != pred_y[i]:
        # print(test_label[i],pred_y[i])
        false_index.append(i)

print(false_index)
# for i in false_index:
#     plt.figure()
#     plt.plot(test_data[i])
#     plt.title('truth: '+ str(test_label[i])+' predict: '+str(pred_y[i]))
#     plt.savefig(str(i)+'.png')
