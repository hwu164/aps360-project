import numpy as np
from sklearn.dummy import DummyClassifier   # ml library
from sklearn.metrics import precision_score, recall_score

"""
refer to
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
"""

# parameters
data_size = 10000
chosen_seed = 82

# generate dummy email data
emails = [f"email {i}" for i in range(0, data_size)]  
rng = np.random.default_rng(seed=chosen_seed)  
labels = rng.choice([0,1], size=data_size, p=[0.95, 0.05])   # 0 for ham, 1 for phishing; 95/5 distribution

# prepare arrays
X = np.array(emails).reshape(-1, 1) # shape should be (N, 1) as sklearn assumes multiple features
y = np.array(labels)

# create stratified dummy classifier
dummy_clf = DummyClassifier(strategy="stratified", random_state=chosen_seed)
dummy_clf.fit(X, y)
y_pred = dummy_clf.predict(X)

# create evaluation metrics
dummy_accuracy = dummy_clf.score(X, y)
dummy_precision = precision_score(y_true=y, y_pred=y_pred, pos_label=1, zero_division=0)
dummy_recall = recall_score(y_true=y, y_pred=y_pred, pos_label=1, zero_division=0)

print(f"Accuracy: {dummy_accuracy*100:.2f}%")
print(f"Precision: {dummy_precision*100:.2f}%")    # true positives/(true positives+false positives) --> out of all emails predicted as phishing, how many were phishing?
print(f"Recall: {dummy_recall*100:.2f}%")          # true positives/(true positives+false negatives) --> how many phishing were caught out of the total phishing amount?