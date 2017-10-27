# coding=utf-8

# write code...

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG,
               outputs=(
                   daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(
                       fmt="%(asctime)s [PID %(process)d] [%(levelname)s] " "%(name)s -> %(message)s")),
                   daiquiri.output.File("errors.log", level=logging.INFO)
               ))


logger = daiquiri.getLogger(__name__)
logger.info("It works and log to stderr by default with color!")
