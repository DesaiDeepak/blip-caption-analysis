import logging
import os
from datetime import datetime


def get_logger(name:str, logs_dir:str="logs")->logging.Logger:
    os.makedirs(logs_dir,exist_ok=True)

    log_filename=os.path.join(logs_dir,f"{datetime.now().strftime('%Y-%m-%d')}.log")

    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        #write logs to file

        file_handler=logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        #Also show logs in terminal

        console_handler=logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter= logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger