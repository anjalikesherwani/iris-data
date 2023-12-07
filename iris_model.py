from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle


iris = datasets.load_iris()
X= iris.data# we only take the first two features.
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_model = SVC()
lin_reg=lin_reg.fit(X_train, y_train)
print("Linear Regression score: ", lin_reg.score(X_test, y_test))
log_reg=log_reg.fit(X_train, y_train)
print("Logistic Regression score: ", log_reg.score(X_test, y_test))
svc_model=svc_model.fit(X_train, y_train)
print("SVC score: ", svc_model.score(X_test, y_test))
pickle.dump(lin_reg,open('lin_model.pkl' , 'wb'))
pickle.dump(log_reg, open('log_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl','wb'))


                                                    