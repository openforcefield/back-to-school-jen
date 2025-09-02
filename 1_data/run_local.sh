#!/bin/bash

######## (MVP) Download and Process SPICE2 Dataset from Zenodo #######
python get_data_spice2.py --data-dir "." 2>&1 | tee log.txt
