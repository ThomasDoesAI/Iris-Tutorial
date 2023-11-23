# load dependencies
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# load the dataset
filepath = "data/iris.data"

# column names to use
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "Class"]

# read the dataset from the URL
dataset = pd.read_csv(filepath, names=names)

# check the first few rows of iris-classification data
print(dataset.head())

# separate the independent and dependent features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Split dataset into random training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# feature standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# training a KNN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# make predictions on the testing data
y_predict = model.predict(X_test)

# check results
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

# save the iris classification model as a pickle file
model_pkl_file = "models/iris_k_neighbors_classifier_model.pkl"

with open(model_pkl_file, "wb") as file:
    pickle.dump(model, file)
