import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from new_dataset import NewsDataset

def read_file(data_file):
    docs = pd.read_csv(data_file)
    text = docs['NEWSCONTENT'].tolist()
    labels_text = docs['NEWSTYPE'].tolist()
    return text, labels_text


def get_dataset(data_file):
    text, labels_text = read_file(data_file)
    #划分数据集
    train_data, test_data, train_labels_text, test_labels_text = train_test_split(text,labels_text,test_size=0.3, random_state=42)
    train_dataset = NewsDataset(train_data, train_labels_text)
    test_dataset = NewsDataset(test_data, test_labels_text)

    return train_dataset, test_dataset








