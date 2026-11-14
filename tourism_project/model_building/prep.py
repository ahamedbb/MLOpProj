# Use the project folder name, then the folder where model building code will be stored.
import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)    # path to the registered tourism.csv inside the data folder
df.drop(columns=["CustomerID"], inplace=True)    # drop the customer identifier column, it is not a predictive feature

target = "ProdTaken"  # name of the column to predict (whether customer purchased the package)
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y    # which variable should stay balanced across the splits?
)

# Explicitly cast object columns to category dtype so XGBoost native categorical handling works seamlessly
for col in Xtrain.select_dtypes(include=['object']).columns:
    Xtrain[col] = Xtrain[col].astype('category')
    Xtest[col] = Xtest[col].astype('category')

# Save the split files into your data directory
OUTPUT_DIR = "tourism_project/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Xtrain.to_csv(os.path.join(OUTPUT_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(OUTPUT_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(OUTPUT_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(OUTPUT_DIR, "ytest.csv"), index=False)

print("Data prepared: train/test splits written with categorical types preserved.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
