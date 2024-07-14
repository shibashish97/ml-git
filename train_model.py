from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from load_data import get_data

def train():
    df = get_data()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train()
    print(f"Model trained with scaling and PCA. Coefficients: {model.coef_}")
