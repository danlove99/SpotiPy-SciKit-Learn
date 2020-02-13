import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('top50.csv', encoding = "ISO-8859-1")

X = data.drop(['ID', 'Track.Name', 'Artist.Name', 'Genre', 'Loudness..dB..', 'Popularity'], axis=1)
y = data['Popularity']

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction
knn.predict([[95, 57, 73, 11, 5, 182, 9, 16]])

# With logistic regression
#logreg = LogisticRegression()
#logreg.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)
#print(metrics.accuracy_score(y_test, y_pred))
