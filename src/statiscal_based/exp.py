import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from feature import *


def read_data():
    data = np.load('data/dataset.npy')
    label = np.load('data/label.npy')
    index = np.logical_and(label != 12, label != 14)  # fix label
    data, label = data[index], label[index]
    return data, label


def feature_extractor(x):
    
    features = [] + basic_info(x)
    features = features + startend(x)
    features = features + absolute_sum_of_changes(x)
    features = features + distribution(x)
    features = features + count_with_mean(x)
    features = features + location(x)
    features = features + cid_ce(x)
    return features


data, label = read_data()

# feature extraction
features = []
for i in data:
    features.append(feature_extractor(i))
train_data, test_data, train_label, test_label = train_test_split(
    features, label, test_size=0.4, random_state=40)
print(np.array(train_data).shape, np.array(train_label).shape)

model = RandomForestClassifier()
model.fit(train_data, train_label)
pred_y = model.predict(test_data)
p, r, f, _ = precision_recall_fscore_support(
    test_label, pred_y, labels=range(13), average="weighted")
print(model.feature_importances_)
print(f"precision: {p}\nrecall: {r}\nf-score: {f}")
