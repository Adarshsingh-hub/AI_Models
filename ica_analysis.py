import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

data = pd.read_csv('./telco_churn.csv')
data ["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ica = FastICA(n_components=3, random_state=42)

X_ica = ica.fit_transform(X_scaled)

print("Independent Components Shape:", X_ica.shape)

plt.scatter(X_ica[:,0], X_ica[:,1])
plt.xlabel("IC1")
plt.ylabel("IC2")
plt.title("ICA Components")
plt.show()