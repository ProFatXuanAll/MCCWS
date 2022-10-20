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
TAG_SET_B_ID = TAG_SET['b']
TAG_SET_E_ID = TAG_SET['e']
TAG_SET_M_ID = TAG_SET['m']
TAG_SET_S_ID = TAG_SET['s']

# Supported datasets.
ALL_DSETS = ['as', 'cityu', 'cnc', 'ctb8', 'msr', 'pku', 'sxu', 'weibo']
