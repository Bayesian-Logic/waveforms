from types import SimpleNamespace
from typing import Iterator, List
import hashlib
import json
import logging

import numpy as np
from sqlite3 import Connection, Cursor


def run_query(conn: Connection, query: str) -> Iterator[SimpleNamespace]:
    """
    Returns an iterator over the query results such that each column is an
    attribute of the returned object.
    """
    curs: Cursor = conn.execute(query)
    colnames: List[str] = [coldesc[0].lower() for coldesc in curs.description]

    for row in curs.fetchall():
        row = SimpleNamespace(**{name: value for (name, value) in zip(colnames, row)})
        yield row


def is_validation_waveform(waveform_file: str, fold: int, num_folds: int):
    return (
        int(hashlib.md5(waveform_file.encode("utf8")).hexdigest(), 16) % num_folds
    ) == fold


def json_to_py(json_cfg):
    """
    Convert a JSON object to a Python object, i.e.
    insead of `j["foo"]["bar"]` we can write `p.foo.bar`
    where `p = json_to_py(j)`.
    """
    return json.loads(json.dumps(json_cfg), object_hook=lambda d: SimpleNamespace(**d))


def downsample(arr: np.ndarray, new_size):
    assert len(arr) >= new_size
    # Create the shrunken array by selecting values at calculated indices
    indices = np.linspace(0, len(arr) - 1, new_size).astype(int)
    return arr[indices]


def update_config(config, param, value):
    """
    `config` is a JSON object whose key in `param` will be updated
    to `value`. Note that the key can be nested for example `a.b.c`.
    This raises an exception if the key is not found.
    """
    keys = param.split(".")
    current_level = config
    for key in keys[:-1]:
        current_level = current_level[key]
    if keys[-1] not in current_level:
        raise ValueError(f"key '{param}' doesn't exist in config.")
    current_level[keys[-1]] = value


def configure_logger():
    # Configure logging with a custom formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Create a handler and set the formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    logging.getLogger().addHandler(handler)

    # Set the desired logging level
    logging.getLogger().setLevel(logging.INFO)
