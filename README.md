# Deep Learning on Waveform Data

This repo can be used to work with waveform data that could be downloaded from the
Segment archive, for example, and train a model to learn the arrival time pick.

## Data

Data can be downloaded with the script `scripts/create_segment_dataset.py`. The default
location of this data is in the `./data` directory.

## Training a model

Install all the requirements by running

    pip install -r requirements.txt

Then you can do a run by invoking the `scripts/trainer.py` script. 

    python scripts/trainer.py

This uses the config file `./config.yaml`. However, you can override any of the parameters on the command line, for example, to do a full run:

    python scripts/trainer.py train.debug=false
