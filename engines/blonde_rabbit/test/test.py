import os
import sys

# Ensure project root is on sys.path so package imports work when running this file directly
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engines.blonde_rabbit.src.model import *
try:
    import pytest
except Exception:
    pytest = None


def test_model_initialization():
    config = Config()
    model_instance = BlondeRabbit(config)
    assert model_instance is not None
    assert isinstance(model_instance, BlondeRabbit)

# for debugger
if __name__ == "__main__":
    test_model_initialization()
    print("All tests passed.")