import logging
import sys

def configure_logging(log_file, level=logging.INFO):

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_file, mode="w")
    ch = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter(
        "%(asctime)s %(processName)s %(levelname)s [%(name)s]: %(message)s"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)

    return root

    # logger.info("This is a sample info")
    # logger.warning("This is a sample warning")
    # logger.error("This is a sample error")