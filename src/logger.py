import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO, # logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR
)
logger = logging.getLogger()
