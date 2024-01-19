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
import os
import sys
import datetime
import calendar
import argparse
import logging

import cx_Oracle
import pandas
import pyarrow
import pyarrow.parquet
import numpy as np

COLUMNS = """orid arid sta chan phase snr amp per delta mb ml ndef
arrival_time timeres auto_arrival_time samprate calib data""".split()


def get_logger():
    logger = logging.getLogger("waveforms")
    # for the first call to a logger we need to setup the handlers
    if not len(logger.handlers):
        logger.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


info = get_logger().info
debug = get_logger().debug


def read_arguments():
    # Read the arguments and parse out the julian date into standard format
    parser = argparse.ArgumentParser(
        prog="create_segment_dataset",
        description="Read waveforms from the segment archive and create a dataset",
    )
    parser.add_argument(
        "start",
        metavar="JDATE",
        nargs=1,
        type=str,
        help="Starting date of dataset (inclusive).",
    )
    parser.add_argument(
        "end",
        metavar="JDATE",
        nargs=1,
        type=str,
        help="Ending date of dataset (not inclusive).",
    )
    parser.add_argument(
        "-w",
        "--window",
        metavar="SECONDS",
        type=int,
        default=100,
        help="Window of waveform to extract around the arrival time in seconds (100 s).",
    )
    parser.add_argument(
        "-c",
        "--channel",
        metavar="NAME",
        type=str,
        default="cb",
        help="Name of channel to download (cb).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.window % 2 != 0:
        raise ValueError(f"Window {args.window} is not a multiple of 2.")

    def jdate_to_epoch(jdate_val):
        datetime_val = datetime.datetime.strptime(jdate_val, "%Y%j")
        return calendar.timegm(datetime_val.timetuple())

    args.start_epoch = jdate_to_epoch(args.start[0])
    args.end_epoch = jdate_to_epoch(args.end[0])
    if args.verbose:
        logger = get_logger()
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    info(
        f"Downloading data from {args.start[0]} to {args.end[0]} ({args.start_epoch} - {args.end_epoch})."
    )
    return args


def main():
    args = read_arguments()
    in_conn = cx_Oracle.connect(os.getenv("ADB_ACCOUNT"))
    debug("Connected to Oracle database.")
    # We will first query all the Origin IDentifiers of the LEB
    # origins and then we will download the waveforms for each of them.
    curs = in_conn.cursor()
    curs.execute(
        f"""select orid from leb.origin where time
        >= {args.start_epoch} and time < {args.end_epoch}"""
    )
    all_orids = [x for x, in curs.fetchall()]
    info(f"{len(all_orids)} origins in date range.")
    data, tot_events, tot_waveforms = [], 0, 0
    for idx, orid in enumerate(all_orids):
        waveforms = fetch_event(curs, orid, args.channel, args.window)
        if len(waveforms) > 0:
            tot_events += 1
            tot_waveforms += len(waveforms)
            data.extend(waveforms)
        if (idx + 1) % (len(all_orids) // 10) == 0:
            info(f"Downloaded {100 * (idx+1) // len(all_orids)} %.")
    data_file = f"{args.channel}-{args.window}-{args.start[0]}-{args.end[0]}.parquet"
    info("Generating parquet file.")
    df = pandas.DataFrame(data=data, columns=COLUMNS)
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    with pyarrow.parquet.ParquetWriter(data_file, table.schema) as writer:
        writer.write_table(table)
    info(
        f"Finished generating dataset in {data_file} with {tot_events}"
        f" events and {tot_waveforms} waveforms."
    )


def fetch_event(curs, orid, channel, window):
    """
    Fetch all the waveforms for the given `orid`.

    We will query each arrival of the orid that has the given `channel`
    waveform using `curs` and read `window` seconds around the
    arrival time.

    The arrivals are returned as a list of rows where each row
    has the columns in the `COLUMNS` global variable.
    """
    debug(f"Fetching event orid {orid}")
    # We join the LEB origin, assoc, and arrival tables to get the main
    # details of each origin and arrival. Then we join this to
    # SEGMENT wftag and wfdisc to get the relevant waveform information.
    # We also optionally join with IDCX arrival in case an automatic
    # arrival exists.
    curs.execute(
        f"""select orid, arid, sta, wf.chan, lass.phase, larr.snr, larr.amp,
        larr.per, lass.delta, lo.mb, lo.ml, lo.ndef, wf.time as start_time,
        wf.endtime as end_time, larr.time as arrival_time, lass.timeres,
        iarr.time as auto_arrival_time,
        wf.nsamp, wf.samprate, wf.calib,
        case when datatype='t4' then '>f4' when datatype='s4' then '>i4' end as dtype,
        wf.dir, wf.dfile, wf.foff
        from leb.origin lo join leb.assoc lass using (orid)
        join leb.arrival larr using (arid, sta)
        join segment.wftag tag on (tagname='orid' and tagid=orid)
        join segment.wfdisc wf using (sta, wfid)
        left join idcx.arrival iarr using (arid, sta)
        where orid=:1 and datatype in ('t4', 's4') and wf.chan = '{channel}'
        and lass.phase in ('P', 'Pn', 'Pg') and larr.snr > 10
        and lass.timedef='d'
        and larr.time between wf.time + {window//2} and wf.endtime - {window//2}
        """,
        [orid],
    )
    waveforms = []
    colnames = [coldesc[0].lower() for coldesc in curs.description]
    for row in curs.fetchall():
        row = {name: value for (name, value) in zip(colnames, row)}
        debug(
            f"Downloading waveform for sta {row['sta']} phase {row['phase']}"
            f" chan {row['chan']} start {row['start_time']} end {row['end_time']} from"
            f" dir {row['dir']} dfile {row['dfile']} foff {row['foff']}"
        )
        path = os.path.join(row["dir"], row["dfile"])
        with open(path, "rb") as fp:
            fp.seek(row["foff"])
            raw_buf = fp.read(row["nsamp"] * 4)  # each sample is 4 bytes
            # Convert the raw data bytes into a numpy array and extract only
            # the relevant `window` samples from it.
            raw_data = np.ndarray(
                shape=(int(row["nsamp"]),), dtype=row["dtype"], buffer=raw_buf
            )
            st = int(
                (row["arrival_time"] - window // 2 - row["start_time"])
                * row["samprate"]
            )
            num = int(window * row["samprate"])
            row["data"] = raw_data[st : st + num].astype(np.float32)
            assert len(row["data"]) == num
            datum = []
            for colname in COLUMNS:
                datum.append(row[colname])
            waveforms.append(datum)
    return waveforms


if __name__ == "__main__":
    main()
