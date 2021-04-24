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

        # Task block
        
        self.ss8 = nn.LSTM(input_size=in_features, num_layers=8, hidden_size=1, dropout=0.8)
        self.ss3 = nn.LSTM(input_size=in_features, num_layers=3, hidden_size=1, dropout=0.8)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]

