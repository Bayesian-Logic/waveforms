# Overview

These scripts are a collection of utilities to collect data as well as Python notebooks
to analyze and build models for specific tasks.

# Scripts

## create_dataset.py

This script reads information about arrivals and waveforms from various IDC tables in
the ADB and then creates a dataset of those arrivals that are associate to LEB events
in a specified time interval. The resulting dataset is stored in a sqlite database file.

In order to run this script the following prerequisites must be met in your environment:

- Python packages numpy and cx_Oracle must be installed.
- Access to the `/archive/ops/seismic/` folder is needed to read the waveforms.
- Environment variable `ADB_ACCOUNT` must have credentials to ADB. For example: user/pass@adb

Example Usage:

    python create_dataset.py -d 365 2022001

The above command will create a dataset of LEB arrivals for `20` days starting from Julian date `2022001` i.e. Jan 1, 2022 (inclusive).

It will create a sqlite db file `2022001-365.db` which can be used by the subsequent steps.

Note that this script prints a lot of errors like the following. This is to do with the inability of the waveform parsing scripts to read the waveforms correctly in all cases. 
However, the script continues to make progress on other waveforms that it can download.

```
Error: no data found for sta BATI elem BATI chan BHZ time 1642354835.25
Exception:
could not broadcast input array from shape (3001) into shape (3000)
Error: no data found for sta SONM elem SONA0 chan SHZ time 1642321478.76
Exception:
could not broadcast input array from shape (3001) into shape (3000)
Error: no data found for sta MDP elem MDP chan BHZ time 1642322133.02
Exception:
could not broadcast input array from shape (6001) into shape (6000)
Error: no data found for sta BDFB elem BDFB chan BHZ time 1642338766.51
.Exception:
cannot convert float NaN to integer
Error: no data found for sta BATI elem BATI chan BHZ time 1642402948.225
Exception:
cannot convert float NaN to integer
Error: no data found for sta BATI elem BATI chan BHZ time 1642439206.45
.Exception:
cannot convert float NaN to integer
Error: no data found for sta EKA elem EKA chan ib time 1642432483.95
```

# Notebooks

## dataset1.ipynb

This file produces a torch datasets for training and validation from the underlying sqlite database.

It assumes that a file `2022001-365.db` has been uploaded in the `waveforms` sub-folder. It then proceeds to produce the dataset files `train_dataset1.data` and `valid_dataset1.data`.

## model7.ipynb

This notebook trains a model from the previously generate dataset files.

It assumes that the two files `train_dataset1.data` and `valid_dataset1.data` are somewhere accessible in the `HOMEDIR` variable. After training it will save a file `model7.pic` with the trained model.

There is some evaluation code as well as some other experiments that were done plus a
visualization of the model layers.

# Helper files

## utils.py

This is a simple script of utilities to read waveforms. It is used primarily by `create_dataset.py`.
