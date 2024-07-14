from train_model import train
from sklearn.metrics import accuracy_score

def evaluate():
    model, X_test, y_test = train()
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate()
    print(f"Model accuracy: {accuracy}")
