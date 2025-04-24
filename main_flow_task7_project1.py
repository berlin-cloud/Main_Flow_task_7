import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset from CSV
iris = pd.read_csv(r'A:\main_Flow_7\iris_dataset.csv')  

# Extract features and target
X = iris.iloc[:, :-1]  
y = iris['target']   
target_names = ['setosa', 'versicolor', 'virginica']  

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Convert to DataFrame for easier plotting
df_reduced = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
df_reduced['Target'] = y

# Scatter plot
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(df_reduced[df_reduced['Target'] == i]['PC1'], df_reduced[df_reduced['Target'] == i]['PC2'], label=target_name)

plt.title('PCA of Iris Dataset (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
