import logging


class Logger(object):
    """
    Helper class for logging.
    Arguments:
        path (str): Path to log file.
    """
    def __init__(self, path, mode='w'):
        self.logger = logging.getLogger()
        self.path = path
        self.mode = mode
        self.setup_file_logger()
        print ('Logging to file: ', self.path)
        
    def setup_file_logger(self):
        hdlr = logging.FileHandler(self.path, self.mode)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        print (message)
        self.logger.info(message)
