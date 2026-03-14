import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("insurance_claims_5000_detailed.csv")

df = df.drop(columns=["Customer_ID"])

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Claim_Approved",axis=1)
y = df["Claim_Approved"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier()

model.fit(X_train,y_train)

joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")
