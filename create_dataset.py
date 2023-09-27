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

# Read waveforms corresponding to LEB+IDCX arrivals for further processing.
# We assume that credentials are stored in the environment variable ADB_ACCOUNT.
import os
import sys
import argparse
import cx_Oracle
import datetime
import calendar
import sqlite3
import json
import numpy as np

import utils

IN_ARRIVALS_QUERY = """
select arid, sta, ia.time as idcx_arr_time, ia.azimuth as idcx_azimuth,
ia.slow as idcx_slow, ia.snr as idcx_snr, ia.amp as idcx_amp, ia.per as idcx_per,
la.time as leb_arr_time, la.azimuth as leb_azimuth,
la.slow as leb_slow, ass.timeres, ass.phase, ass.delta, orid, ori.mb, ori.ml, ori.ndef
from idcx.arrival ia join leb.arrival la using (arid, sta) 
join leb.assoc ass using (arid, sta) join leb.origin ori using (orid)
where ia.time between :1 and :2 and ass.timedef='d'
"""

OUT_TABLE_CREATE = """
CREATE TABLE arrival_waveform (
  arid INTEGER,
  sta TEXT,
  elem TEXT,
  chan TEXT,
  phase TEXT,
  snr REAL,
  amp REAL,
  per REAL,
  delta REAL,
  timeres REAL,
  orid INTEGER,
  mb REAL,
  ml REAL,
  ndef INTEGER,
  time_difference REAL,
  sampling_rate INTEGER,
  data BLOB
)
"""


def main():
    args = read_arguments()
    conn = cx_Oracle.connect(os.getenv("ADB_ACCOUNT"))
    sitechan = read_sta_channel_names(args, conn)
    site = read_sites(args, conn)
    arrivals = read_arrivals(args, conn)
    out_conn = sqlite3.connect(f"{args.jdate}-{args.days}.db")
    add_waveforms(args, conn, sitechan, site, arrivals, out_conn)
    conn.close()
    out_conn.close()


def read_arguments():
    # Read the arguments and parse out the julian date into standard format
    parser = argparse.ArgumentParser(
        prog="create_dateset",
        description="Read waveforms corresponding to LEB+IDCX arrivals and make a dataset",
    )
    parser.add_argument(
        "jdate", metavar="JDATE", nargs=1, type=str, help="Date of arrivals to query."
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=30,
        help="The size of the window around the arrival time in seconds (default 30).",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=1,
        help="The number of days of data to download (default 1).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    args.jdate = args.jdate[0]
    args.datetime = datetime.datetime.strptime(args.jdate, "%Y%j")
    args.epoch = calendar.timegm(args.datetime.timetuple())
    print(f"Processing jdate {args.jdate} date {args.datetime} epoch {args.epoch}")
    return args


# Returns a dictionary of station names -> channels. We restrict ourselves to
# vertical component channels only. And in case of duplicates we use the channel
# that was created most recently.
def read_sta_channel_names(args, conn):
    curs = conn.cursor()
    curs.execute(
        """select sta, chan from static.sitechan where descrip like '%vertical%' and
:1 >= ondate and (offdate=-1 or :2 < offdate) order by ondate, lddate""",
        [args.jdate, args.jdate],
    )
    sitechan = dict(curs.fetchall())
    curs.close()
    print(
        f"{len(sitechan)} stations mapped to vertical component channel names e.g. NV01 -> {sitechan['NV01']}"
    )
    return sitechan


def read_sites(args, conn):
    curs = conn.cursor()
    curs.execute(
        f"""select sta, statype, refsta from static.site where ondate <= {args.jdate}
and (offdate=-1 or offdate > {args.jdate}) order by ondate, lddate"""
    )
    rows = curs.fetchall()
    dict_rows = utils.add_column_names(curs, rows)
    curs.close()
    return {row["sta"]: row for row in dict_rows}


# Each arrival row has the following fields -- arid, sta, idcx_arr_time,
# idcx_azimuth, idcx_slow, leb_arr_time, leb_azimuth, leb_slow,
# timeres, phase, delta, orid, mb, ml, ndef
def read_arrivals(args, conn):
    curs = conn.cursor()
    curs.execute(IN_ARRIVALS_QUERY, [args.epoch, args.epoch + args.days * 24 * 60 * 60])
    rows = curs.fetchall()
    dict_rows = utils.add_column_names(curs, rows)
    curs.close()
    print(f"{len(rows)} arrival rows fetched.")
    print(f"first arrival row {dict_rows[0]}")
    print(f"last arrival row {dict_rows[-1]}")
    return dict_rows


# adds a waveform to each arrival row and store in the sqlite database
def add_waveforms(args, conn, sitechan, site, arrivals, out_conn):
    # Create a table to store the data.
    out_curs = out_conn.cursor()
    out_curs.execute(OUT_TABLE_CREATE)
    # Read the waveforms corresponding to each arrival and store in the output
    # table.
    curs = conn.cursor()
    unk_sta = set()
    unk_sitechan = set()
    for cnt, arr in enumerate(arrivals):
        if args.verbose:
            print(f"Processing Arrival: {arr}")
        sta = arr["sta"]
        if sta not in site:
            if sta not in unk_sta:
                print(f"Warning: unknown station {sta}")
                unk_sta.add(sta)
            continue
        # Use the reference station code for array stations.
        siterow = site[sta]
        if siterow["statype"] == "ss":
            elem = sta
        elif siterow["statype"] == "ar":
            elem = siterow["refsta"]
        else:
            print(f"Fatal: unknown statype {siterow['statype']} for sta {sta}")
            sys.exit(1)
        if elem not in sitechan:
            if elem not in unk_sitechan:
                print(
                    f"Warning: element {elem} of station {sta} has no vertical channels."
                )
                unk_sitechan.add(elem)
            continue
        time_difference = arr["leb_arr_time"] - arr["idcx_arr_time"]
        # If the correct arrival time is too close to the edge of the window then skip this
        # data sample.
        if abs(time_difference) > (args.window - 1):
            continue
        data, sr = utils.get_waveform_data(
            curs,
            elem,
            sitechan[elem],
            arr["idcx_arr_time"] - args.window,
            arr["idcx_arr_time"] + args.window,
        )
        if data is None:
            print(
                f"Error: no data found for sta {sta} elem {elem} chan {sitechan[elem]} time {arr['idcx_arr_time']}"
            )
            continue
        if args.verbose:
            print(f"Sampling rate {sr} len data = {len(data)} data = {data}")
        out_curs.execute(
            """INSERT INTO arrival_waveform (arid, sta, elem, chan, phase, snr, amp, per, delta, timeres,
orid, mb, ml, ndef, time_difference, sampling_rate, data) values 
(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                arr["arid"],
                sta,
                elem,
                sitechan[elem],
                arr["phase"],
                arr["idcx_snr"],
                arr["idcx_amp"],
                arr["idcx_per"],
                arr["delta"],
                # Note: we want the timeres w.r.t. the idcx arrival time
                arr["idcx_arr_time"] - arr["leb_arr_time"] + arr["timeres"],
                arr["orid"],
                arr["mb"],
                arr["ml"],
                arr["ndef"],
                time_difference,
                sr,
                json.dumps(data.tolist()),
            ),
        )
        # commit every 100 arrivals
        if (cnt + 1) % 100 == 0:
            out_conn.commit()
            if not args.verbose:
                print(".", end="", flush=True)
    out_conn.commit()
    curs.close()
    out_curs.close()


if __name__ == "__main__":
    main()
