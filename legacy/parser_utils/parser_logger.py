import logging
import os
from datetime import datetime
# import inspect
class CustomFileHandler(logging.FileHandler):
    def __init__(self):
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)
        log_filename = os.path.join(log_directory, f"parserapp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        super().__init__(log_filename)

def setup_logger(verbose=False):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    # formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(funcName)s(): %(message)s")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s")
    # Create a console handler with INFO level
    console_handler = logging.StreamHandler()
    if verbose:
        stream_log_level = logging.DEBUG
    else:
        stream_log_level = logging.INFO
    console_handler.setLevel(stream_log_level)
    console_handler.setFormatter(formatter)

    # Create a custom file handler with DEBUG level
    file_handler = CustomFileHandler()
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)
    logger.debug("logger initialized!")

# Set up the logger when this module is imported
setup_logger()
