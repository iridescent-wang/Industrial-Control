import os
import pandas as pd
import numpy as np


def preprocessing():
    IN_DIR = 'dataset/pcap/'
    OUT_DIR = 'dataset/'

    label_num, label_lst = 0, []
    train_df, test_df = pd.DataFrame(), pd.DataFrame()

    for file in os.listdir(IN_DIR):
        if file == 'smtp.pcap_Flow.csv':
            continue

        df = pd.read_csv(IN_DIR + file).iloc[:, 7:].replace([np.inf, -np.inf], 0).dropna()
        df['Label'] = [label_num] * df.shape[0]
        label_lst.append(file.split('.')[0])
        label_num += 1

        # c_train_df = df.sample(frac=0.7, random_state=42)
        # c_test_df = df[~df.index.isin(c_train_df.index)]
        train_size = int(df.shape[0] * 0.7)
        c_train_df = df[:train_size]
        c_test_df = df[train_size:]

        train_df = pd.concat([train_df, c_train_df])
        test_df = pd.concat([test_df, c_test_df])

    train_df.sample(frac=1.0, random_state=42).to_csv(OUT_DIR + 'train.csv', index=False)
    test_df.sample(frac=1.0, random_state=42).to_csv(OUT_DIR + 'test.csv', index=False)

    return label_lst
