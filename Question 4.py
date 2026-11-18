import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

unique, count = np.unique(y, return_counts=True)
print("Class Count: ")
for i in range(len(unique)):
    print(f"Class {unique[i]}: {count[i]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

prob = model.predict_proba(X_test)[:,1]
threshold = [0.3,0.5,0.7]

for th in threshold:
    y_thresh= (prob>= th).astype(int)
    count = np.sum(y_thresh ==1)
    accuracy =accuracy_score(y_test, y_thresh)
    print(f"Threshold: {th}")
    print(f"Class 1 count: {count}")
    print(f"Accuracy: {accuracy}")

def sigmoid(z):
    return 1/(1+np.exp(-z))

val = model.decision_function(X_test)
sigVal = sigmoid(val)
probVal = model.predict_proba(X_test)[:,1]

print(f"Sigmoid Verification: {np.allclose(sigVal, probVal)}")
