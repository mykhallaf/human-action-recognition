import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


print("Loading data")
df = pd.read_csv('action_dataset.csv')


X = df.drop('label', axis=1)
y = df['label']


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)


print("Selecting most efficient features")

selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"Original feature count: {X_train.shape[1]}")
print(f"Selected feature count: {X_train_selected.shape[1]}")


models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "ANN (Neural Network)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000),
    "SVM": SVC()
}


best_model_name = ""
best_accuracy = 0
best_y_pred = []

print("\n---Model Results---")
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")


    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_y_pred = y_pred


print(f"\nBest Model: {best_model_name} with {best_accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual Action')
plt.xlabel('Predicted Action')
plt.show()


print(f"Saving {best_model_name}...")
with open('action_model.pkl', 'wb') as f:
    pickle.dump(models[best_model_name], f)
print("Model saved as 'action_model.pkl'")

print("Saving Feature Selector")
with open('selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
print("Selector saved as 'selector.pkl'")
