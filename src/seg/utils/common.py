from types import SimpleNamespace
from sqlite3 import Connection, Cursor
from typing import Iterator, List
import hashlib
import json

import numpy as np


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
    assert(len(arr) >= new_size)
    # Create the shrunken array by selecting values at calculated indices
    indices = np.linspace(0, len(arr) - 1, new_size).astype(int)
    return arr[indices]
