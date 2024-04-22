import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self,input_size, squeeze_size, device='cpu'):
        super().__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 740),
            nn.ReLU(),
            nn.Linear(740, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Linear(80, 40))
        
        self.decoder = nn.Sequential(
            nn.Linear(40, 80),
            nn.ReLU(),
            nn.Linear(80, 160),
            nn.ReLU(),
            nn.Linear(160, 320),
            nn.ReLU(),
            nn.Linear(320, 640),
            nn.ReLU(),
            nn.Linear(640, 740),
            nn.ReLU(),
            nn.Linear(740, input_size),
            nn.Sigmoid())
        
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
