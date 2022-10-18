import os
from typing import Final

PROJECT_ROOT: Final[str] = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))

# Data related path.
DATA_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'data')
DOWNLOAD_DATA_PATH: Final[str] = os.path.join(DATA_PATH, 'download')
RAW_DATA_PATH: Final[str] = os.path.join(DATA_PATH, 'raw')
PREPROCESS_DATA_PATH: Final[str] = os.path.join(DATA_PATH, 'preprocess')

# Experiment related path.
EXP_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'exp')

# Log related path.
LOG_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'log')

# BMES tagset.
TAG_SET = {'b': 0, 'e': 1, 'm': 2, 's': 3, 'pad': 4}

# Supported datasets.
ALL_DSETS = ['as', 'cityu', 'ctb8', 'msr', 'pku', 'sxu', 'weibo']
