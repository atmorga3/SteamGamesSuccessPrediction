import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class steamGamesDataset(Dataset):
    def __init__(self, lhs_dataframe, rhs_dataframe):
        self.lhs_df = lhs_dataframe
        self.rhs_df = rhs_dataframe
        # self.dataset = pd.concat([lhs_dataframe, rhs_dataframe], axis=1)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        inp_features = self.lhs_df.iloc[index].to_numpy()
        value = self.rhs_df.iloc[index].to_numpy()
        return inp_features, value