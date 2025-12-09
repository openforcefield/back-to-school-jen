#!/bin/bash

python ../convert_pt_to_offxml.py \
    --checkpoint my-smee-fit/force-field-epoch-990.pt \
    --offxml-in ../../forcefields/openff_unconstrained-2.2.1.offxml \
    --offxml-out openff-2.2.1-smee-spice.offxml
