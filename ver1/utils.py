# Copyright (c) 2023, Bayesian-Logic
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Some functions are modified from: https://github.com/hofmanradek/ML/blob/master/wavedataset/src/read_wfdisc.py

import math
import os
import sys
import numpy as np


BYTES_PER_SAMPLE = {"t4": 4, "s3": 3, "s4": 4}


# Extract column names from a cursor and convert the list of rows into a list
# of dictionaries with the column name added to each row.
def add_column_names(cursor, rows):
    colnames = [coldesc[0].lower() for coldesc in cursor.description]
    return [{name: value for (name, value) in zip(colnames, row)} for row in rows]


def convert_raw_data(raw_data, datatype, mult=1.0):
    """
    converst raw data read from a wfdisc file to numpy floats
    :param raw_data:
    :param datatype:
    :return:
    """
    bytes_per_sample = BYTES_PER_SAMPLE.get(datatype, None)
    n = int(raw_data.size / float(bytes_per_sample))

    if datatype == "t4":
        ret = np.ndarray(shape=(n,), dtype=">f4", buffer=raw_data)
    elif datatype == "s3":  # this crazy operation is from Dima Bobrov
        buf = np.hstack(
            (np.zeros((n, 1), dtype="uint8"), raw_data.reshape(n, -1))
        ).ravel()
        ret = np.ndarray(shape=(n,), dtype=">i4", buffer=buf)
        ret[np.where(ret >= 0x800000)] -= 0x1000000
    elif datatype == "s4":
        ret = np.ndarray(shape=(n,), dtype=">i4", buffer=raw_data)
    else:
        print("Unknown datatype, exiting...")
        sys.exit(1)

    return ret * mult


def get_wfdisc_entries(cursor, sta, chan, t1, t2):
    """
    compiles a query for getting a wfdisc entry
    :param sta: station
    :param chan: channel
    :param t1: start of desired interval
    :param t2: end of desired interval
    :return: query for select of all relevant wfdisc files
    Name     Null?    Type
    -------- -------- ------------
    STA      NOT NULL VARCHAR2(6)
    CHAN     NOT NULL VARCHAR2(8)
    TIME     NOT NULL FLOAT(53)
    WFID     NOT NULL NUMBER(10)
    CHANID            NUMBER(8)
    JDATE             NUMBER(8)
    ENDTIME           FLOAT(53)
    NSAMP             NUMBER(8)
    SAMPRATE          FLOAT(24)
    CALIB             FLOAT(24)
    CALPER            FLOAT(24)
    INSTYPE           VARCHAR2(6)
    SEGTYPE           VARCHAR2(1)
    DATATYPE          VARCHAR2(2)
    CLIP              VARCHAR2(1)
    DIR               VARCHAR2(64)
    DFILE             VARCHAR2(32)
    FOFF              NUMBER(10)
    COMMID            NUMBER(10)
    LDDATE            DATE
    """
    assert t1 < t2
    # This query considers two possible scenarios..
    # CASE1 the interval starts in a wfdisc file and ends in another - start part, possible middle part and end part
    # CASE2 the whole interval is in a single wfdisc file
    query = f"""SELECT * FROM idcx.wfdisc WHERE  sta='{sta}' AND chan='{chan}' AND (
  -- 
  ((time<={t1} AND {t1}<endtime AND {t2}>endtime) OR
   (time>{t1} AND {t2}>=endtime) OR
   (time<{t2} AND {t2}<=endtime AND time>{t1}))
  OR
  ((time<={t1} AND {t1}<=endtime) AND (time<={t2} AND {t2}<=endtime))
) ORDER BY sta, chan, time, foff """
    rows = cursor.execute(query)
    return add_column_names(cursor, rows)


def read_waveforms_from_files(t1, t2, wfdict, calib, sr):
    """
    reads waveform data from wfdisc files
    :return index: position of the first data item relative to t1
    :return nsamp: number of returned samples
    :return subdata: vector of data starting at index
    """
    path = os.path.join(wfdict["dir"], wfdict["dfile"])
    offset = wfdict["foff"]
    samprate = sr  # wfdict['samprate']
    time = wfdict["time"]
    endtime = wfdict["endtime"]
    datatype = wfdict["datatype"]
    calib_fact = wfdict["calib"] if calib else 1.0  # calibrate if NOT raw
    bytes_per_sample = BYTES_PER_SAMPLE.get(datatype, None)

    # we set start and end time in this particular wfdisc file w.r.t t1 and t2
    tstart = time if t1 <= time else t1
    tend = (
        endtime if t2 >= endtime else t2 - 1 / samprate
    )  # the last sample is the real end...

    # print(offset, wfdict['datatype'], path)

    # offset in wfdisc entry is in bytes
    with open(path, "rb") as f:
        # we seek to the beginning of current station/channel
        # print('reading wfdisc: ', path)

        # we calculate sample start and sample end for this particular wfdisc file
        startsample = int(
            (tstart - time) * samprate
        )  # int(round((tstart - time) * samprate))
        endsample = int(
            (tend - time) * samprate
        )  # int(round((tend - time) * samprate))
        nsamp = (
            endsample - startsample + 1
        )  # this must be calculated, sometimes we read not of all samples
        f.seek(offset)
        # read raw data into numpy uint8 ndarray
        # print('= number of samples: %d' % nsamp)
        raw_data = np.fromfile(
            f, dtype="uint8"
        )  # , count=int(round(nsamp * bytes_per_sample)))

        # convert to floats given its particular type
        subdata = convert_raw_data(raw_data, datatype, mult=calib_fact)[
            startsample : endsample + 1
        ]

        index = int((tstart - t1) * samprate)
        return index, nsamp, subdata


def get_samprates(wfdicts):
    """
    from all entries in wfdict extracts samprates
    :param wfdicts:
    :return:
    """
    return [wfd["samprate"] for wfd in wfdicts]


def get_waveform_data(cursor, sta, chan, t1, t2, calib=True):
    """
    crates a numpy array with samples
    :param sta: station
    :param chan: channel
    :param t1: start of desired interval
    :param t2: end of desired interval
    :param cursor: (oracle) db cursor
    :return: numpy array with samples
    """
    wfdicts = get_wfdisc_entries(cursor, sta, chan, t1, t2)
    # average samprate

    try:
        sr = round(np.mean(get_samprates(wfdicts)))
        data = np.zeros(int(math.ceil((t2 - t1) * sr)))
        data_masks = np.ones(int(math.ceil((t2 - t1) * sr)))
        # print('SAMPRATE: %3.2f Hz  %s' % (sr, get_samperates(wfdicts)))

        for wfdict in wfdicts:
            # now we must calculate which samples will be occupied by the retrieved data from each wfdisc file
            index, nsamp, subdata = read_waveforms_from_files(t1, t2, wfdict, calib, sr)
            # put data chunk in place
            data[index : index + nsamp] = subdata
            data_masks[index : index + nsamp] = 0  # unmasks those data we have
        return data, sr
    except Exception as e:
        print("Exception:")
        print(e)
        return None, None
