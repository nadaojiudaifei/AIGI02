import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from aigi02.config import tiny_config
from aigi02.model import AIGIModel

@pytest.fixture
def model():
    torch.set_num_threads(1);torch.manual_seed(17)
    return AIGIModel(tiny_config()).eval()
