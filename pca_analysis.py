import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('./telco_churn.csv')

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna()

features = ["tenure", "MonthlyCharges", "TotalCharges"]

X = data[features]

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

#visualization
plt.plot(pca.explained_variance_ratio_, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.title("PCA Variance")
plt.show()