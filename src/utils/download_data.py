import io
import logging
import os
import zipfile
from typing import Final

import requests
from tqdm import tqdm

import src.vars

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_SIZE: Final[int] = 1024


def create_data_folder() -> None:
  """Create data folder if not exist."""
  # Create data root folder.
  if not os.path.exists(src.vars.DATA_PATH):
    os.makedirs(src.vars.DATA_PATH)
  # Create download data folder.
  if not os.path.exists(src.vars.DOWNLOAD_DATA_PATH):
    os.makedirs(src.vars.DOWNLOAD_DATA_PATH)
  # Create raw data folder.
  if not os.path.exists(src.vars.RAW_DATA_PATH):
    os.makedirs(src.vars.RAW_DATA_PATH)


def create_download_progress_bar(desc: str, total_bytes: int) -> tqdm:
  """Create download progress bar.

  Source code: https://stackoverflow.com/a/62113293/9908486
  """
  return tqdm(
    desc=desc,
    total=total_bytes,
    unit='iB',
    unit_scale=True,
    unit_divisor=DOWNLOAD_CHUNK_SIZE,
  )


def download_binary_file(desc: str, download_file_name: str, url: str) -> None:
  """Download binary file.

  Source code: https://stackoverflow.com/a/62113293/9908486
  """
  download_file_path = os.path.join(src.vars.DOWNLOAD_DATA_PATH, download_file_name)

  # Make sure data folder exist and is structured as we want.
  create_data_folder()

  # Do nothing if dataset is already downloaded.
  if os.path.exists(download_file_path):
    logger.info(f'Skip downloading binary file: {download_file_name} is already downloaded.')
    return

  logger.info('Start downloading binary file.')
  with requests.get(url, stream=True) as response, open(download_file_path, 'wb') as download_file:
    download_progress_bar = create_download_progress_bar(
      desc=desc,
      total_bytes=int(response.headers.get('content-length', 0)),
    )
    for data in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
      size = download_file.write(data)
      download_progress_bar.update(size)
    download_progress_bar.close()
  logger.info('Finish downloading binary file.')


def download_SIGHAN_2005_bakeoff() -> None:
  """Download SIGHAN 2005 bakeoff datasets and extract from zip file."""
  # Download dataset.
  download_binary_file(
    desc='downloading SIGHAN 2005 bakeoff',
    download_file_name='icwb2-data.zip',
    url='http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip',
  )

  # zip file structure mapping.
  txt_file_mapping = [
    # Train files.
    ('icwb2-data/training/as_training.utf8', 'as_train.txt'),
    ('icwb2-data/training/cityu_training.utf8', 'cityu_train.txt'),
    ('icwb2-data/training/msr_training.utf8', 'msr_train.txt'),
    ('icwb2-data/training/pku_training.utf8', 'pku_train.txt'),
    # Test files.
    ('icwb2-data/gold/as_testing_gold.utf8', 'as_test.txt'),
    ('icwb2-data/gold/cityu_test_gold.utf8', 'cityu_test.txt'),
    ('icwb2-data/gold/msr_test_gold.utf8', 'msr_test.txt'),
    ('icwb2-data/gold/pku_test_gold.utf8', 'pku_test.txt'),
  ]

  is_all_file_extracted = True
  for _, output_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name)):
      is_all_file_extracted = False
      break

  if is_all_file_extracted:
    logger.info('Skip extracting raw data: SIGHAN 2005 bakeoff raw data are already extracted.')
    return

  logger.info('Start extracting raw data.')
  with zipfile.ZipFile(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'icwb2-data.zip'), 'r') as input_zipfile:
    for input_txt_file_path, output_txt_file_name in txt_file_mapping:
      logger.info(f'Start extracting {output_txt_file_name}.')
      with io.TextIOWrapper(input_zipfile.open(input_txt_file_path, 'r'), encoding='utf-8') as input_txt_file:
        data = input_txt_file.read()
        with open(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name), 'w') as output_txt_file:
          output_txt_file.write(data)
      logger.info(f'Finish extracting {output_txt_file_name}.')
  logger.info('Finish extracting raw data.')
