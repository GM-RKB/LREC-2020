import os
import logging

LOGGER = logging.getLogger(__name__)  # Every log will use the module name
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.DEBUG)


class Configuration(object):
    """NNet configurations"""


CONFIG = Configuration()

# Preprocessing parameters
CONFIG.padding = "🔳"
CONFIG.max_input_len = 40
CONFIG.inverted = True

# For prediction probabilities
CONFIG.allowed_threshold = .95

# Parameters for evaluation:
CONFIG.k = 1
CONFIG.DATA_FILES_PATH = os.path.abspath("./nnet/data")
CONFIG.TRAINED_FILES_PATH = os.path.abspath("./nnet_models/")
CONFIG.DATA_FILES_FULL_PATH = os.path.expanduser(CONFIG.DATA_FILES_PATH)
CONFIG.TRAINED_FILES_FULL_PATH = os.path.expanduser(CONFIG.TRAINED_FILES_PATH)

CONFIG.ALLOWED_CHARS_FILE_NAME = os.path.join(CONFIG.DATA_FILES_FULL_PATH, "allowed_chars_sm.json")


CONFIG.SAVED_MODEL_FILE_NAME = os.path.join(
    CONFIG.TRAINED_FILES_FULL_PATH, "gm_rkb_nnet_fixer_GMRKB&Wiki7_sm_e22.h5")
