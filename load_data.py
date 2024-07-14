import pandas as pd
from sklearn.datasets import load_iris

def get_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

if __name__ == "__main__":
    df = get_data()
    print(df.head())
