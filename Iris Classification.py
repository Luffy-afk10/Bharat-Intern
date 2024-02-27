from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()
X = iris.data  
y = iris.target  


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_tr, y_tr)


y_pred = rf_classifier.predict(X_te)


accuracy = accuracy_score(y_te, y_pred)
print("Accuracy:", accuracy)


print("report",classification_report(y_te, y_pred, target_names=iris.target_names))
