import torch
import torch.nn as nn

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class Baseline(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(Baseline, self).__init__()
        
        self.num_layers = 2
        self.hidden_size = 160
        
        # Task block
        
        self.lstm = nn.LSTM(in_features, self.hidden_size, self.num_layers, dropout=0.8, batch_first = True)
        self.fc_ss8 = nn.Linear(hidden_size, 8)
        self.fc_ss3 = nn.Linear(hidden_size, 3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        
        h0 = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        h1 = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        c1 = torch.zeros(self.num_layers, x.size[0], self.hidden_size)
        
        ss8, _ = self.lstm(x, (h0, c0))
        ss8 = ss8[:, -1, :]
        ss8 = self.fc_ss8(ss8)
        
        ss3, _ = self.lstm(x, (h1, c1))
        ss3 = ss3[:, -1, :]
        ss3 = self.fc_ss3(ss3)
        return [ss8, ss3]

