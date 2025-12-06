# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import Necessary Libraries and Load Data 

2.Split Dataset into Training and Testing Sets 

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy 

5.Generate Confusion Matrix


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Mohamed Aathil M
RegisterNumber:  25008235
*/
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)



## Output:
<img width="777" height="316" alt="Screenshot 2025-12-06 084239" src="https://github.com/user-attachments/assets/9de90dc3-5bca-44a6-832f-4f95f370b67e" />

<img width="187" height="19" alt="Screenshot 2025-12-06 084314" src="https://github.com/user-attachments/assets/8226009b-aad5-4b4e-b8a0-c3e016c77883" />

<img width="176" height="97" alt="Screenshot 2025-12-06 084323" src="https://github.com/user-attachments/assets/23627959-4311-4f95-bc5e-5869d0c8586b" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
