import pandas as pd

def prepare_data(X):
    labels = X["fire"]
    data = X.drop(columns="fire")

    data["Date"] = pd.to_datetime(data["Date"])
    labels = labels.astype("category")
    return data, labels