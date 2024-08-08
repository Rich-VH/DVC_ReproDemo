import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop('Cabin', axis=1)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked'])
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df = df.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/raw/titanic.csv')
    df = preprocess_data(df)
    df.to_csv('data/processed/titanic_processed.csv', index=False)
