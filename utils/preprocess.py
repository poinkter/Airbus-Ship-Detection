import pandas as pd
from sklearn.model_selection import train_test_split

def undersample_df(df):
    # select a random 10% of all data samples to be with 0 ships 
    zero_ships = df[df['ships']==0].sample(n=int(len(df)*0.1))

    nonzero_ships = df[df['ships'] != 0]
    return pd.concat((zero_ships, nonzero_ships), axis=0)

def preprocess(path, valid_size=0.3):
    train = pd.read_csv(path)

    # Convert 'EncodedPixels' column to binary labels (1 for non-empty, 0 for empty)
    train['ships'] = train['EncodedPixels'].apply(lambda x: 1 if isinstance(x, str) else 0)

    # Count number of ships per image
    unique_train = train.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

    # Drop 'ships' column for future merge
    train.drop(['ships'], axis=1, inplace=True)

    train_ids, valid_ids = train_test_split(unique_train,
                                        test_size=valid_size,
                                        stratify=unique_train['ships'])

    # Fill NaN values with empty string
    train['EncodedPixels'] = train['EncodedPixels'].fillna("") 

    train_df = pd.merge(train, train_ids)
    valid_df = pd.merge(train, valid_ids)

    # undersample samples with 0 ships
    train_df = undersample_df(train_df)
    valid_df = undersample_df(valid_df)

    return (train_df, valid_df)