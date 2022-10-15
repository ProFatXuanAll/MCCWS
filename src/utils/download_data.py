import io
import logging
import os
import zipfile
import shutil
from typing import Final

import requests
from tqdm import tqdm
import pyunpack

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


def download_file(desc: str, download_file_name: str, url: str) -> None:
  """Download file with either binary or text mode.

  mode can only be 'w' or 'wb', where 'w' stands for text mode and 'wb' stands for binary mode.

  Source code: https://stackoverflow.com/a/62113293/9908486
  """
  download_file_path = os.path.join(src.vars.DOWNLOAD_DATA_PATH, download_file_name)

  # Make sure data folder exist and is structured as we want.
  create_data_folder()

  # Do nothing if dataset is already downloaded.
  if os.path.exists(download_file_path):
    logger.info(f'Skip downloading file: {download_file_name} is already downloaded.')
    return

  logger.info('Start downloading file.')
  with requests.get(url, stream=True) as response, open(download_file_path, 'wb') as download_file:
    download_progress_bar = create_download_progress_bar(
      desc=desc,
      total_bytes=int(response.headers.get('content-length', 0)),
    )
    for data in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
      size = download_file.write(data)
      download_progress_bar.update(size)
    download_progress_bar.close()
  logger.info('Finish downloading file.')


def download_SIGHAN_2005_bakeoff() -> None:
  """Download SIGHAN 2005 bakeoff datasets and extract from zip file."""
  # Download dataset.
  download_file(
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
        with open(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name), 'w', encoding='utf-8') as output_txt_file:
          output_txt_file.write(data)
      logger.info(f'Finish extracting {output_txt_file_name}.')
  logger.info('Finish extracting raw data.')


def download_NLPCC_2016_Weibo() -> None:
  """Download NLPCC 2016 Weibo datasets.

  To get the answer of test set, one need to apply it directly to FudanNLP lab.
  See https://github.com/FudanNLP/NLPCC-WordSeg-Weibo
  """
  # Download dataset.
  download_file(
    desc='downloading NLPCC 2016 Weibo training set',
    download_file_name='nlpcc2016-word-seg-train.dat',
    url='https://raw.githubusercontent.com/FudanNLP/NLPCC-WordSeg-Weibo/main/datasets/nlpcc2016-word-seg-train.dat',
  )
  download_file(
    desc='downloading NLPCC 2016 Weibo dev set',
    download_file_name='nlpcc2016-wordseg-dev.dat',
    url='https://raw.githubusercontent.com/FudanNLP/NLPCC-WordSeg-Weibo/main/datasets/nlpcc2016-wordseg-dev.dat',
  )
  download_file(
    desc='downloading NLPCC 2016 Weibo test set',
    download_file_name='nlpcc2016-wordseg-test.dat',
    url='https://raw.githubusercontent.com/FudanNLP/NLPCC-WordSeg-Weibo/main/datasets/nlpcc2016-wordseg-test.dat',
  )

  # zip file structure mapping.
  txt_file_mapping = [
    ('nlpcc2016-word-seg-train.dat', 'weibo_train.txt'),
    ('nlpcc2016-wordseg-dev.dat', 'weibo_dev.txt'),
    ('nlpcc2016-wordseg-test.dat', 'weibo_test.txt'),
  ]

  is_all_file_renamed = True
  for _, output_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name)):
      is_all_file_renamed = False
      break

  if is_all_file_renamed:
    logger.info('Skip renaming raw data: NLPCC 2016 Weibo raw data are already renamed.')
    return

  logger.info('Start renaming raw data.')
  for input_txt_file_name, output_txt_file_name in txt_file_mapping:
    logger.info(f'Start renaming {output_txt_file_name}.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, input_txt_file_name), 'r', encoding='utf-8') as input_txt_file:
      data = input_txt_file.read()
      with open(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name), 'w', encoding='utf-8') as output_txt_file:
        output_txt_file.write(data)
    logger.info(f'Finish renaming {output_txt_file_name}.')
  logger.info('Finish renaming raw data.')


def download_SIGHAN_2008_bakeoff_SXU() -> None:
  """Download SIGHAN 2008 bakeoff SXU datasets and extract from rar file.

  This is so hard to find.
  I am not able to find SXU from their original website, or from any lab of ShanXi University.
  """
  # Download dataset.
  download_file(
    desc='downloading SIGHAN 2008 bakeoff SXU dataset',
    download_file_name='backoff2008.rar',
    url='https://github.com/tjzhifei/tjzhifei.github.com/raw/master/resources/backoff2008.rar',
  )

  # zip file structure mapping.
  txt_file_mapping = [
    # Train files.
    ('中文分词评测测试+训练语料/中文分词评测训练语料（山西大学提供）/训练语料（528250词，Unicode格式）.txt', 'sxu_train.txt'),
    # Test files.
    ('中文分词评测测试+训练语料/中文分词评测测试语料（山西大学提供）/测试语料答案（Unicode格式）.txt', 'sxu_test.txt'),
  ]

  is_all_file_extracted = True
  for _, output_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name)):
      is_all_file_extracted = False
      break

  if is_all_file_extracted:
    logger.info('Skip extracting raw data: SIGHAN 2008 bakeoff SXU raw data are already extracted.')
    return

  logger.info('Start extracting raw data.')
  pyunpack.Archive(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'backoff2008.rar')).extractall(src.vars.DOWNLOAD_DATA_PATH)
  for input_txt_file_path, output_txt_file_name in txt_file_mapping:
    logger.info(f'Start renaming {output_txt_file_name}.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, input_txt_file_path), 'r', encoding='utf-16') as input_txt_file:
      data = input_txt_file.read()
      with open(os.path.join(src.vars.RAW_DATA_PATH, output_txt_file_name), 'w', encoding='utf-8') as output_txt_file:
        output_txt_file.write(data)
    logger.info(f'Finish renaming {output_txt_file_name}.')
  shutil.rmtree(os.path.join(src.vars.DOWNLOAD_DATA_PATH, '中文分词评测测试+训练语料'))
  logger.info('Finish extracting raw data.')
