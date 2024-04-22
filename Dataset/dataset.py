import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class steamGamesDataset(Dataset):
    def __init__(self, lhs_dataframe, rhs_dataframe, device='cpu'):
        self.lhs_df = lhs_dataframe
        self.rhs_df = rhs_dataframe
        # self.dataset = pd.concat([lhs_dataframe, rhs_dataframe], axis=1)
        
    def __len__(self):
        return len(self.lhs_df)
    
    def __getitem__(self, index):
        inp_features = torch.tensor(self.lhs_df.iloc[index]).to(torch.float32)
        value = torch.tensor(self.rhs_df.iloc[index]).to(torch.float32)
        return inp_features, value