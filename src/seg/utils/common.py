from types import SimpleNamespace
from typing import Iterator, List
import hashlib
import json
import logging

import numpy as np
from sqlite3 import Connection, Cursor
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_mae_by_snr(val_data, file_name, q=10):
    """
    Computes plots of Mean Absolute Error grouped by SNR buckets and
    saves it to `file_name`.

    val_data:
    A pandas dataframe with columns `snr` and `pred_err`
    file_name:
    File to store the plot.
    q:
    Number of quantiles. 10 for deciles, 4 for quartiles, etc.
    Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    Also, see documentation for pandas.qcut,
    https://pandas.pydata.org/docs/reference/api/pandas.qcut.html
    """
    # create SNR buckets by percentile
    val_data["snr_bucket"], snr_bins = pd.qcut(val_data["snr"], q=q, retbins=True)

    # compute the absolute error
    val_data["abs_err"] = np.abs(val_data.pred_err)

    # Compute MAE for each snr bucket
    mae_per_bucket = val_data.groupby("snr_bucket", observed=False)["abs_err"].mean()

    # Get midpoints of SNR buckets for x-axis labels
    snr_midpoints = [((a + b) / 2) for a, b in zip(snr_bins[:-1], snr_bins[1:])]

    # Plot the MAE per snr bucket with custom x-axis labels
    fig, ax = plt.subplots()
    mae_per_bucket.plot(
        kind="bar",
        ax=ax,
        xlabel="SNR Bucket",
        ylabel="Mean Absolute Error",
        title="MAE per SNR Bucket",
    )

    # Customize x-axis labels with one decimal value
    ax.set_xticklabels([f"{mid:.1f}" for mid in snr_midpoints], rotation=45, ha="right")

    fig.savefig(file_name)
    plt.close(fig)


def plot_err_by_quantiles(err_vals, file_name, num_quantiles=60, trim=0.01):
    """
    trim - Fraction of data to be dropped. We will drop the extreme quantiles
    """
    fig = plt.figure()

    # Define quantiles (e.g., quartiles)
    quantiles = np.linspace(trim / 2, 1 - trim / 2, num_quantiles + 1)
    quantile_values = np.quantile(err_vals, quantiles)

    # Create histogram with quantile buckets
    plt.hist(err_vals, bins=quantile_values)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title("Val Error")

    fig.savefig(file_name)
    plt.close(fig)


def plot_dfx_err_overlay(val_data, file_name, q=10):
    """
    Computes plots of Mean Absolute Error (MAE) and the difference between
    auto_arrival_time and arrival_time overlaid on each other,
    grouped by SNR buckets, and saves it to `file_name`.

    val_data:
    A pandas dataframe with columns `snr`, `pred_err`, `arrival_time`,
    and `auto_arrival_time`.
    file_name:
    File to store the plot.
    q:
    Number of quantiles. 10 for deciles, 4 for quartiles, etc.
    Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    Also, see documentation for pandas.qcut,
    https://pandas.pydata.org/docs/reference/api/pandas.qcut.html
    """
    # Only consider rows with an auto_arrival_time and make a copy of the dataframe.
    val_data = val_data.dropna(subset=["auto_arrival_time"])[
        ["pred_err", "snr", "arrival_time", "auto_arrival_time"]
    ].copy()

    # Compute the absolute error of Deep Learning method
    val_data["Deep Learning"] = np.abs(val_data.pred_err)

    # Compute the difference between auto_arrival_time and arrival_time
    val_data["dfx_err"] = val_data["auto_arrival_time"] - val_data["arrival_time"]
    val_data["DFX"] = np.abs(val_data.dfx_err)

    # Create SNR buckets by percentile
    val_data["snr_bucket"], snr_bins = pd.qcut(val_data["snr"], q=q, retbins=True)

    # Compute MAE for each SNR bucket
    mae_per_bucket = val_data.groupby("snr_bucket", observed=False)[
        ["Deep Learning", "DFX"]
    ].mean()

    # Get midpoints of SNR buckets for x-axis labels
    snr_midpoints = [((a + b) / 2) for a, b in zip(snr_bins[:-1], snr_bins[1:])]

    # Plot the MAE per SNR bucket with custom x-axis labels
    fig, ax = plt.subplots()

    mae_per_bucket.plot(
        kind="bar",
        ax=ax,
        xlabel="Upper value of SNR bucket",
        ylabel="Mean Absolute Error",
        title="MAE per SNR Bucket",
    )

    ax.set_xticklabels(
        [f"{high:.1f}" for high in snr_bins[1:]], rotation=0, ha="center"
    )

    fig.tight_layout()
    fig.savefig(file_name)
    plt.close(fig)


def plot_err_by_range(err_vals, file_name, num_bins=20, low=-1, high=1):
    bins = np.linspace(low, high, num_bins + 1)

    fig, ax = plt.subplots()
    # Create histogram with quantile buckets
    n, bins, patches = ax.hist(err_vals, bins=bins)

    # Centering count over the corresponding bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(n, bin_centers):
        ax.text(x, count, str(int(count)), ha="center", va="bottom")

    # Adding a dotted vertical line parallel to the y-axis
    ax.axvline(x=0, linestyle="dotted", color="black")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Count")
    plt.title("Val Error")

    fig.savefig(file_name)
    plt.close(fig)
