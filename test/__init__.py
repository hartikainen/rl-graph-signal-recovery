import pathlib
import logging
logging.getLogger('tensorflow').disabled = True

pathlib.Path('./tmp').mkdir(parents=True, exist_ok=True)
