import torch
from torch.utils.data import Dataset
import pandas as pd

def getTybaltDatasets(path, val_percent = 0.1):
    rnaseq_df = pd.read_table(path, index_col=0)

    rnaseq_test_df = rnaseq_df.sample(frac=val_percent)
    rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

    dataset_val = TybaltDataset(rnaseq_test_df)
    dataset_train = TybaltDataset(rnaseq_train_df)

    return dataset_train, dataset_val


class TybaltDataset(Dataset):

    def __init__(self, dataframe):
        super(TybaltDataset, self).__init__()
        self.data = torch.tensor(dataframe.values)
        print("Loaded data of size: ", self.data.size())

    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, idx):
        return self.data[idx].type(torch.FloatTensor)
        

