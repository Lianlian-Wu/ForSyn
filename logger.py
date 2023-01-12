import logging
import sys
import time

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')
    
    # filename="log/{}.txt".format(time.strftime("%Y%m%d_%H:%M:%S", time.localtime()))

    file_handler = logging.FileHandler("log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle=logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger
