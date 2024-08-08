import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def split_data(df, test_size, random_state):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train, n_estimators, random_state):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    df = pd.read_csv('data/processed/titanic_processed.csv')
    X_train, X_test, y_train, y_test = split_data(df, params['split']['test_size'], params['split']['random_state'])
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train, params['model']['n_estimators'], params['model']['random_state'])
    joblib.dump(model, 'model.pkl')
