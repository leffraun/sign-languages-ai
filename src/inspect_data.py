import pandas as pd

df = pd.read_csv("../landmarks/landmarks.csv")

print(df.head())
print()
print("Shape:", df.shape)
print()
print("Columns:")
print(df.columns)
X = df.drop("label", axis=1)
y = df["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_Test, y_train, y_test = train_test_split(
    x,
    y,
    test_Size=0.2,
    random_state=42,
    stratify=y
)
