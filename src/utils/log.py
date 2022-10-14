import logging
import os

import src.vars


def create_log_folder() -> None:
  """Create log folder if not exist."""
  # Create log root folder.
  if not os.path.exists(src.vars.LOG_PATH):
    os.makedirs(src.vars.LOG_PATH)


def setup_log_to_file(exp_name: str, log_level: str = 'DEBUG'):
  """Record experiment logs in file.

  This function must always be called before any logging event is called.
  See https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
  """
  # Make sure log folder exist.
  create_log_folder()

  # assuming log_level is bound to the string value obtained from the command line argument.
  # Convert to upper case to allow the user to specify --log=DEBUG or --log=debug
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')

  log_file_name = f'{os.path.join(src.vars.LOG_PATH, exp_name)}.log'

  logging.basicConfig(
    encoding='utf-8',
    filemode='w',
    filename=log_file_name,
    level=numeric_level,
  )
