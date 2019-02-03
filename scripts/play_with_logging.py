"""
defines a command-line utility for the creation of the database
"""

import sys
import pandas as pd
import logging
from logging.handlers import SocketHandler
from pythonjsonlogger import jsonlogger
from async_v20 import OandaClient
import asyncio
import time

def test_logging():
    """tests logging"""

    # log to a file


    # put root logging directly into cutelog
    logger_root = logging.getLogger('Root logger')
    logger_root.setLevel(logging.DEBUG)
    socket_handler = SocketHandler('127.0.0.1', 19996)  # default listening address
    logger_root.addHandler(socket_handler)

    # get logger for async_v20 and set level
    logger_async_v20 = logging.getLogger('async_v20')
    logger_async_v20.setLevel(logging.DEBUG)

    # put async_v20 logging output directly into cutelog
    logger_async_v20.addHandler(socket_handler)

    # define a streaming handler
    lsh = logging.StreamHandler()
    lf_str_txtfile = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    lsh.setFormatter(logging.Formatter(lf_str_txtfile))

    # add streaming handler to async_v20
    logger_async_v20.addHandler(lsh)

    # add streaming handler to root
    logger_root.addHandler(lsh)

    # put async_v20 logging output into my.log
    lfh = logging.FileHandler('my.log')
    lfh.setFormatter(logging.Formatter(lf_str_txtfile))  # use the same formatter as used for the streaming handler
    logger_async_v20.addHandler(lfh)

    # put async_v20 logging output into my_json.log
    lfh = logging.FileHandler('my_json.log')
    lfh.setFormatter(jsonlogger.JsonFormatter(reserved_attrs=[]))
    logger_async_v20.addHandler(lfh)

    # put root logging output into my.log
    lfh = logging.FileHandler('my.log')
    lfh.setFormatter(logging.Formatter(lf_str_txtfile))  # use the same formatter as used for the streaming handler
    logger_root.addHandler(lfh)

    # put root logging output into my_json.log
    lfh = logging.FileHandler('my_json.log')
    lfh.setFormatter(jsonlogger.JsonFormatter(reserved_attrs=[]))
    logger_root.addHandler(lfh)


    client = OandaClient()
    loop = asyncio.get_event_loop()
    rsp = loop.run_until_complete(client.close_all_trades())

    logger_async_v20.info("df")
    logger_root.info("ddf")

    i = 0
    while True:
        logger_async_v20.info(f'Hello world {i:04d}!')
        logger_root.info(f'Hello world {i:04d}!')
        i = i + 1
        time.sleep(2)

    pass



def main(argv):
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 300)
    test_logging()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
