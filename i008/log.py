import logging
import sys


def get_logger(lvl=logging.DEBUG, logger_name='default', elk_host=None, elk_port=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(lvl)

    if elk_host and elk_port:
        import logstash
        logstash_handler = logstash.LogstashHandler(elk_host, int(elk_port))
        logstash_handler.setLevel(lvl)
        logger.addHandler(logstash_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stdout_handler.setLevel(lvl)
    stdout_handler.setFormatter(stdout_formatter)

    # hdlr = logging.FileHandler('nwfilelog.log')
    # hdlr.setFormatter(stdout_formatter)
    # hdlr.setLevel(lvl)

    logger.addHandler(stdout_handler)
    # logger.addHandler(hdlr)

    return logger

# logger = get_logger()
