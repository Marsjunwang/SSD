import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

if __name__ == "__main__":
    d2l.DATA_HUB['banana-detection'] = (
        d2l.DATA_URL + 'banana-detection.zip','5de26c8fce5ccdea9f91267273464dc968d20d72')