import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm

df = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_columns = df.columns[:-1]
df['Symptoms'] = df[symptom_columns].values.tolist()
df = df[['Symptoms', 'Disease']]
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s for s in x if isinstance(s, str) and s.strip()])

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['Symptoms'])
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("model/disease_model.pkl", "wb") as f:
    pickle.dump((model, mlb), f)
