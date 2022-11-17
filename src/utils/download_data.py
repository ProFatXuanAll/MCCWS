import io
import logging
import os
import re
import shutil
import tarfile
import zipfile
from typing import Final

import pyunpack
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

  # Create http header to authenticate ourselve as users.
  # See https://stackoverflow.com/a/67809908
  headers = requests.utils.default_headers()
  headers.update({'User-Agent': 'My User Agent 1.0'})
  with requests.get(url, headers=headers, stream=True) as response, open(download_file_path, 'wb') as download_file:
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
  for _, out_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name)):
      is_all_file_extracted = False
      break

  if is_all_file_extracted:
    logger.info('Skip extracting raw data: SIGHAN 2005 bakeoff raw data are already extracted.')
    return

  logger.info('Start extracting raw data.')

  with zipfile.ZipFile(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'icwb2-data.zip'), 'r') as input_zipfile:
    for in_txt_file_path, out_txt_file_name in txt_file_mapping:
      logger.info(f'Start extracting {out_txt_file_name}.')

      with io.TextIOWrapper(input_zipfile.open(in_txt_file_path, 'r'), encoding='utf-8') as in_txt_file:
        data = in_txt_file.read()
        with open(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name), 'w', encoding='utf-8') as out_txt_file:
          out_txt_file.write(data)

      logger.info(f'Finish extracting {out_txt_file_name}.')

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
    ('nlpcc2016-word-seg-train.dat', 'nlpcc_train.txt'),
    ('nlpcc2016-wordseg-dev.dat', 'nlpcc_dev.txt'),
    ('nlpcc2016-wordseg-test.dat', 'nlpcc_test.txt'),
  ]

  is_all_file_renamed = True
  for _, out_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name)):
      is_all_file_renamed = False
      break

  if is_all_file_renamed:
    logger.info('Skip renaming raw data: NLPCC 2016 Weibo raw data are already renamed.')
    return

  logger.info('Start renaming raw data.')
  for in_txt_file_name, out_txt_file_name in txt_file_mapping:
    logger.info(f'Start renaming {out_txt_file_name}.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, in_txt_file_name), 'r', encoding='utf-8') as in_txt_file:
      data = in_txt_file.read()
      with open(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name), 'w', encoding='utf-8') as out_txt_file:
        out_txt_file.write(data)
    logger.info(f'Finish renaming {out_txt_file_name}.')
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
  for _, out_txt_file_name in txt_file_mapping:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name)):
      is_all_file_extracted = False
      break

  if is_all_file_extracted:
    logger.info('Skip extracting raw data: SIGHAN 2008 bakeoff SXU raw data are already extracted.')
    return

  logger.info('Start extracting raw data.')

  pyunpack.Archive(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'backoff2008.rar')).extractall(src.vars.DOWNLOAD_DATA_PATH)
  for in_txt_file_path, out_txt_file_name in txt_file_mapping:
    logger.info(f'Start renaming {out_txt_file_name}.')

    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, in_txt_file_path), 'r', encoding='utf-16') as in_txt_file:
      data = in_txt_file.read()
      with open(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name), 'w', encoding='utf-8') as out_txt_file:
        out_txt_file.write(data)

    logger.info(f'Finish renaming {out_txt_file_name}.')

  shutil.rmtree(os.path.join(src.vars.DOWNLOAD_DATA_PATH, '中文分词评测测试+训练语料'))
  logger.info('Finish extracting raw data.')


def download_CTB8() -> None:
  """Download CTB8 dataset.

  Source: https://wakespace.lib.wfu.edu/handle/10339/39379
  """
  # Download dataset.
  download_file(
    desc='downloading CTB8 dataset',
    download_file_name='LDC2013T21.tgz',
    url='https://wakespace.lib.wfu.edu/bitstream/handle/10339/39379/LDC2013T21.tgz',
  )

  # File structure mapping.
  # Test file are defined by CTB8.0 official.
  test_file_range = list(range(1, 44)) + list(range(144, 170)) + list(range(900, 932)) + [
    1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148
  ]

  is_all_file_extracted = True
  for out_txt_file_name in ['ctb8_train.txt', 'ctb8_test.txt']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, out_txt_file_name)):
      is_all_file_extracted = False
      break

  if is_all_file_extracted:
    logger.info('Skip extracting raw data: CTB8 raw data are already extracted.')
    return

  logger.info('Start extracting raw data.')

  # Extract CTB8 from tgz file.
  tarfile.open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'LDC2013T21.tgz'), 'r').extractall(src.vars.DOWNLOAD_DATA_PATH)
  data_folder_path = os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'ctb8.0', 'data', 'segmented')

  output_train_txt_file = open(os.path.join(src.vars.RAW_DATA_PATH, 'ctb8_train.txt'), 'w', encoding='utf-8')
  output_test_txt_file = open(os.path.join(src.vars.RAW_DATA_PATH, 'ctb8_test.txt'), 'w', encoding='utf-8')

  for input_file_name in sorted(os.listdir(data_folder_path)):
    logger.info(f'Start extracting {input_file_name}.')

    file_id = int(re.match(r'chtb_(\d{4})', input_file_name)[1])
    in_txt_file = open(os.path.join(data_folder_path, input_file_name), 'r', encoding='utf-8')
    raw_xml = in_txt_file.read()
    in_txt_file.close()

    # Remove XML tags.
    # We do not use XML parser since some data is not valid XML.
    raw_xml = re.sub(r'<DOC>', r'', raw_xml)
    raw_xml = re.sub(r'<DOCNO>.*</DOCNO>', r'', raw_xml)
    raw_xml = re.sub(r'<DOCTYPE[^>]*>.*</DOCTYPE>', r'', raw_xml)
    raw_xml = re.sub(r'<DATE_TIME>.*</DATE_TIME>', r'', raw_xml)
    raw_xml = re.sub(r'<DATETIME>.*</DATETIME>', r'', raw_xml)
    raw_xml = re.sub(r'<DATE>.*</DATE>', r'', raw_xml)
    raw_xml = re.sub(r'<DOCID>.*</DOCID>', r'', raw_xml)
    raw_xml = re.sub(r'<HEADER>.*</HEADER>', r'', raw_xml)
    raw_xml = re.sub(r'<HEADER>', r'', raw_xml)
    raw_xml = re.sub(r'</HEADER>', r'', raw_xml)
    raw_xml = re.sub(r'<BODY>', r'', raw_xml)
    raw_xml = re.sub(r'<HEADLINE>', r'', raw_xml)
    raw_xml = re.sub(r'<S ID=\w+>', r'', raw_xml)
    raw_xml = re.sub(r'</S>', r'', raw_xml)
    raw_xml = re.sub(r'<TEXT>', r'', raw_xml)
    raw_xml = re.sub(r'<TURN>', r'', raw_xml)
    raw_xml = re.sub(r'<P>', r'', raw_xml)
    raw_xml = re.sub(r'</P>', r'', raw_xml)
    raw_xml = re.sub(r'</TURN>', r'', raw_xml)
    raw_xml = re.sub(r'</TEXT>', r'', raw_xml)
    raw_xml = re.sub(r'</HEADLINE>', r'', raw_xml)
    raw_xml = re.sub(r'</BODY>', r'', raw_xml)
    raw_xml = re.sub(r'<ENDTIME>.*</ENDTIME>', r'', raw_xml)
    raw_xml = re.sub(r'<END_TIME>.*</END_TIME>', r'', raw_xml)
    raw_xml = re.sub(r'</DOC>', r'', raw_xml)
    raw_xml = re.sub(r'<seg id="\w+">', r'', raw_xml)
    raw_xml = re.sub(r'</seg>', r'', raw_xml)
    raw_xml = re.sub(r'<segment id="\w+" start="[^"]+" end="[^"]+">', r'', raw_xml)
    raw_xml = re.sub(r'</segment>', r'', raw_xml)
    raw_xml = re.sub(r'<su id=\w+>', r'', raw_xml)

    sents = []
    for sent in re.split(r'\n+', raw_xml):
      sent = sent.strip()
      # Discard empty lines.
      if not sent:
        continue
      # Discard closing hints.
      if sent in ['（ 完 ）', '完', 'ＥＭＰＴＹ']:
        continue
      # Discard lines consist entirely of non-words.
      if re.match(r'^\W+$', sent):
        continue
      sents.append(sent)

    for sent in sents:
      if file_id in test_file_range:
        output_test_txt_file.write(sent + '\n')
      else:
        output_train_txt_file.write(sent + '\n')

    logger.info(f'Finish extracting {input_file_name}.')
  shutil.rmtree(os.path.join(src.vars.DOWNLOAD_DATA_PATH, 'ctb8.0'))

  output_train_txt_file.close()
  output_test_txt_file.close()

  logger.info('Finish extracting raw data.')


def download_CNC() -> None:
  """Download CNC dataset.

  We download from hankcs.
  Source: https://github.com/hankcs/multi-criteria-cws/tree/master/data/other
  """
  # Download dataset.
  download_file(
    desc='downloading CNC training set',
    download_file_name='cnc_train.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/cnc/train.txt',
  )
  download_file(
    desc='downloading CNC dev set',
    download_file_name='cnc_dev.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/cnc/dev.txt',
  )
  download_file(
    desc='downloading CNC test set',
    download_file_name='cnc_test.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/cnc/test.txt',
  )

  is_all_file_copied = True
  for split in ['train', 'dev', 'test']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, f'cnc_{split}.txt')):
      is_all_file_copied = False
      break

  if is_all_file_copied:
    logger.info('Skip parsing raw data: CNC raw data are already parsed.')
    return

  logger.info('Start parsing raw data.')
  for split in ['train', 'dev', 'test']:
    logger.info(f'Start parsing cnc_{split}.txt.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, f'cnc_{split}.txt'), 'r', encoding='utf-8') as in_txt_file:
      lines = in_txt_file.readlines()
    with open(os.path.join(src.vars.RAW_DATA_PATH, f'cnc_{split}.txt'), 'w', encoding='utf-8') as out_txt_file:
      for line in lines:
        words = []
        for word_with_pos in re.split(r'\s+', line.strip()):
          word = re.sub(r'/[a-z]+$', r'', word_with_pos)
          words.append(word)
        sent = ' '.join(words)
        out_txt_file.write(f'{sent}\n')
    logger.info(f'Finish parsing cnc_{split}.txt.')
  logger.info('Finish parsing raw data.')


def download_CTB6() -> None:
  """Download CTB6 dataset.

  We download from hankcs.
  Source: https://github.com/hankcs/multi-criteria-cws/tree/master/data/other
  """
  # Download dataset.
  download_file(
    desc='downloading CTB6 training set',
    download_file_name='ctb6_train.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/ctb/ctb6.train.seg',
  )
  download_file(
    desc='downloading CTB6 dev set',
    download_file_name='ctb6_dev.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/ctb/ctb6.dev.seg',
  )
  download_file(
    desc='downloading CTB6 test set',
    download_file_name='ctb6_test.txt',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/ctb/ctb6.test.seg',
  )

  is_all_file_renamed = True
  for split in ['train', 'dev', 'test']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, f'ctb6_{split}.txt')):
      is_all_file_renamed = False
      break

  if is_all_file_renamed:
    logger.info('Skip renaming raw data: CTB6 raw data are already renamed.')
    return

  logger.info('Start renaming raw data.')
  for split in ['train', 'dev', 'test']:
    logger.info(f'Start renaming ctb6_{split}.txt.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, f'ctb6_{split}.txt'), 'r', encoding='utf-8') as in_txt_file:
      data = in_txt_file.read()
    with open(os.path.join(src.vars.RAW_DATA_PATH, f'ctb6_{split}.txt'), 'w', encoding='utf-8') as out_txt_file:
      out_txt_file.write(data)
    logger.info(f'Finish renaming ctb6_{split}.txt.')
  logger.info('Finish renaming raw data.')


def download_UD() -> None:
  """Download Universal Dependency dataset.

  We download from  hankcs.
  Source: https://github.com/hankcs/multi-criteria-cws/tree/master/data/other
  """
  # Download dataset.
  download_file(
    desc='downloading UD training set',
    download_file_name='ud_train.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/udc/train.conll',
  )
  download_file(
    desc='downloading UD dev set',
    download_file_name='ud_dev.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/udc/dev.conll',
  )
  download_file(
    desc='downloading UD test set',
    download_file_name='ud_test.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/udc/test.conll',
  )

  is_all_file_copied = True
  for split in ['train', 'dev', 'test']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, f'ud_{split}.txt')):
      is_all_file_copied = False
      break

  if is_all_file_copied:
    logger.info('Skip parsing raw data: UD raw data are already parsed.')
    return

  logger.info('Start parsing raw data.')
  for split in ['train', 'dev', 'test']:
    logger.info(f'Start parsing ud_{split}.conll.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, f'ud_{split}.conll'), 'r', encoding='utf-8') as in_txt_file:
      lines = in_txt_file.readlines()
    with open(os.path.join(src.vars.RAW_DATA_PATH, f'ud_{split}.txt'), 'w', encoding='utf-8') as out_txt_file:
      words = []
      for line in lines:
        line = line.strip()
        # Sentence are separated by empty line.
        if not line and words:
          sent = ' '.join(words)
          out_txt_file.write(f'{sent}\n')
          words = []
        else:
          word = re.split(r'\s+', line)[1]
          words.append(word)

      if words:
        sent = ' '.join(words)
        out_txt_file.write(f'{sent}\n')
    logger.info(f'Finish parsing ud_{split}.conll.')
  logger.info('Finish parsing raw data.')


def download_WTB() -> None:
  """Download WTB dataset.

  We download from hankcs.
  Source: https://github.com/hankcs/multi-criteria-cws/tree/master/data/other
  """
  # Download dataset.
  download_file(
    desc='downloading WTB training set',
    download_file_name='wtb_train.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/wtb/train.conll',
  )
  download_file(
    desc='downloading WTB dev set',
    download_file_name='wtb_dev.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/wtb/dev.conll',
  )
  download_file(
    desc='downloading WTB test set',
    download_file_name='wtb_test.conll',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/wtb/test.conll',
  )

  is_all_file_copied = True
  for split in ['train', 'dev', 'test']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, f'wtb_{split}.txt')):
      is_all_file_copied = False
      break

  if is_all_file_copied:
    logger.info('Skip parsing raw data: WTB raw data are already parsed.')
    return

  logger.info('Start parsing raw data.')
  for split in ['train', 'dev', 'test']:
    logger.info(f'Start parsing wtb_{split}.conll.')
    with open(os.path.join(src.vars.DOWNLOAD_DATA_PATH, f'wtb_{split}.conll'), 'r', encoding='utf-8') as in_txt_file:
      lines = in_txt_file.readlines()
    with open(os.path.join(src.vars.RAW_DATA_PATH, f'wtb_{split}.txt'), 'w', encoding='utf-8') as out_txt_file:
      words = []
      for line in lines:
        line = line.strip()
        # Sentence are separated by empty line.
        if not line and words:
          sent = ' '.join(words)
          out_txt_file.write(f'{sent}\n')
          words = []
        else:
          word = re.split(r'\s+', line)[1]
          words.append(word)

      if words:
        sent = ' '.join(words)
        out_txt_file.write(f'{sent}\n')
    logger.info(f'Finish parsing wtb_{split}.conll.')
  logger.info('Finish parsing raw data.')


def download_ZX() -> None:
  """Download ZX dataset.

  We download from hankcs.
  Source: https://github.com/hankcs/multi-criteria-cws/tree/master/data/other
  """
  # Download dataset.
  download_file(
    desc='downloading ZX training set',
    download_file_name='train.zhuxian.wordpos',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/zx/train.zhuxian.wordpos',
  )
  download_file(
    desc='downloading ZX dev set',
    download_file_name='dev.zhuxian.wordpos',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/zx/dev.zhuxian.wordpos',
  )
  download_file(
    desc='downloading ZX test set',
    download_file_name='test.zhuxian.wordpos',
    url='https://raw.githubusercontent.com/hankcs/multi-criteria-cws/master/data/other/zx/test.zhuxian.wordpos',
  )

  is_all_file_copied = True
  for split in ['train', 'dev', 'test']:
    if not os.path.exists(os.path.join(src.vars.RAW_DATA_PATH, f'zx_{split}.txt')):
      is_all_file_copied = False
      break

  if is_all_file_copied:
    logger.info('Skip parsing raw data: ZX raw data are already parsed.')
    return

  logger.info('Start parsing raw data.')
  for split in ['train', 'dev', 'test']:
    logger.info(f'Start parsing {split}.zhuxian.wordpos.')
    with open(
      os.path.join(src.vars.DOWNLOAD_DATA_PATH, f'{split}.zhuxian.wordpos'), 'r', encoding='utf-8'
    ) as in_txt_file:
      lines = in_txt_file.readlines()
    with open(os.path.join(src.vars.RAW_DATA_PATH, f'zx_{split}.txt'), 'w', encoding='utf-8') as out_txt_file:
      for line in lines:
        words = []
        for word_with_pos in re.split(r'\s+', line.strip()):
          word = re.sub(r'_[A-Z]+$', r'', word_with_pos)
          words.append(word)
        sent = ' '.join(words)
        out_txt_file.write(f'{sent}\n')
    logger.info(f'Finish parsing {split}.zhuxian.wordpos.')
  logger.info('Finish parsing raw data.')


def download_liwenzhu() -> None:
  """Download dataset provided by liwenzhu.

  Currently not sure to use it or not.
  Function is left empty

  Source: https://github.com/liwenzhu/corpusZh
  """
  pass


def download_all() -> None:
  """Download all datasets."""
  download_SIGHAN_2005_bakeoff()
  download_SIGHAN_2008_bakeoff_SXU()
  download_NLPCC_2016_Weibo()
  download_CTB6()
  download_CTB8()
  download_CNC()
  download_UD()
  download_WTB()
  download_ZX()


if __name__ == '__main__':
  # Show logs.
  logging.basicConfig(level=logging.INFO)

  # Download all datasets.
  download_all()
