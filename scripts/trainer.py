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
import yaml
import logging
import argparse

# add the source code to the python path
sys.path.insert(1, "./src")
from seg.utils.train import do_train
from seg.utils.common import update_config, configure_logger


def main():
    parser = argparse.ArgumentParser(description="Update config parameters")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file"
    )
    parser.add_argument(
        "params",
        nargs="*",
        help='List of parameters to update in the format "section.parameter=value"',
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp.read())

    for param in args.params:
        if "=" not in param or param.count("=") != 1:
            print(
                f"Invalid input: {param}. Please provide parameters in the format 'section.parameter=value'."
            )
            sys.exit(1)
        key, value = param.split("=")
        update_config(cfg, key, yaml.safe_load(value))

    data, model = do_train(cfg)


if __name__ == "__main__":
    configure_logger()
    main()
