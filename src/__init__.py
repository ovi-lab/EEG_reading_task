# import config
# import analysis
# import tools

import logging 

if not logging.getLogger(__name__).hasHandlers:
    logger =  logging.getLogger(__name__)
    handler =  logging.StreamHandler()
    formatter =  logging.Formatter(
          "%(levelname)-8s :: %(name)-16s :: %(message)-s"
    )

    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)