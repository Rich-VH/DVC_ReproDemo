import pandas as pd
import yaml

def load_data(url):
    return pd.read_csv(url)

if __name__ == "__main__":
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    url = params['data_url']
    df = load_data(url)
    df.to_csv('data/raw/titanic.csv', index=False)
