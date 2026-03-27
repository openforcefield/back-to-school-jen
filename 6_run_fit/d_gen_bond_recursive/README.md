# FF-D: Generalized Bond Recursive (incomplete)

Early experimental fit. No run scripts (`submit.sh`, `log.txt`) are preserved in this directory.
Only the output `final-force-field.offxml` remains.

The final force field contains Angles and ProperTorsions only — Bond parameters are absent,
suggesting either an Angles-only fit or an early run where the source OFFXML lacked a Bonds section.

Note: For context, bond lower bounds in all other fits are `k ≥ 0`, `length ≥ 0`.
This will differ in `d2_*` fits.
