#!/bin/bash

python ../convert_pt_to_offxml.py --checkpoint "my-smee-fit/force-field-epoch-220.pt" \
                                  --offxml-in "../../4_make_offxmls/d_gen_bond_recursive/openff-2.3.0-gen_bond_atom_rec1.offxml" \
				  --offxml-out "final-force-field.offxml"
