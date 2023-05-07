import logging
import logging.config

import yaml


def get_logger():
    with open("./logger.yaml", "r") as stream:
        loggers_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(loggers_config)
    return logging.getLogger(name="app")


logger = get_logger()
