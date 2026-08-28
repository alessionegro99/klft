# Shared comparison provenance

`first_stage_manifest.tsv` is the exact 15-ensemble input to the qbig
first-stage plotting job. `qbig/plot_first_stage.slurm` generated compact PDFs
without transferring raw histories.

`generate_inputs.py` retains the deterministic historical production
generator. `provenance/cadence25/manifest.tsv` is the accepted production
manifest; its ensemble directories are now grouped under the three line
folders. `provenance/` also keeps the superseded cadence-20 campaign, legacy
pilot inputs, and historical qbig wrappers. Those records are not part of the
active continuum selection.
