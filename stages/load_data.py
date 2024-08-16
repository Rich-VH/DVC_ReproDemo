import pandas as pd
import ssl
import yaml

ssl._create_default_https_context = ssl._create_unverified_context

def load_data(url):
    return pd.read_csv(url)

if __name__ == "__main__":
    with open('params.yaml') as f:
        params = yaml.safe_load(f)
    
    url = params['data_url']
    df = load_data(url)
    df.to_csv('data/raw/titanic.csv', index=False)
