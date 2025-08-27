#!/bin/bash

# Download / process data
mkdir -p data
#python tasks/get_data.py --data-dir "data"
python tasks/get_data.py --data-dir "/dfs9/dmobley-lab/openff-bts"
