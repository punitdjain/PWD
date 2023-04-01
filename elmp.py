import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier, ELMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#importing the dataset
dataset = pd.read_csv("Dataset.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1:].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
lb=preprocessing.LabelBinarizer()
lb.fit(y_train)
elm_model = ELMClassifier(n_hidden=50, alpha=0.1, rbf_width=1.0, activation_func='tanh', activation_args=None, user_components=None, regressor=None, binarizer=lb, random_state=None) 
elm_model.fit(x_train, y_train)
predicted = elm_model.predict(x_test)
#probs = elm_model.predict_proba(x_test)
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print("Confusion matrix:\n{}\n".format(conf_mat))
# plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_mat, annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="RdPu")
plt.title("Confusion matrix of Extreme Learning Machine", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)
plt.savefig("static/img/cmelm.png", bbox_inches="tight")
plt.figure(figsize=(12, 8))
elm_fpr, elm_tpr, elm_thresholds = roc_curve(y_test, elm_model.predict(x_test))
print(elm_fpr, elm_tpr, elm_thresholds)
plt.plot(elm_fpr, elm_tpr, label="ELM (AUC = {:1.4f})".format(roc_auc_score(y_test, elm_model.predict(x_test))))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve", fontsize=16)
plt.legend(loc="lower right")
plt.savefig("static/img/rocelm.png", bbox_inches="tight")
plt.show()





