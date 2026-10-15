# Fixed mG Nt

Immutable input copies and provenance. Raw histories remain on qbig at the original manifest paths.

[Immutable manifest](manifest.json) · [Simulation build](build_provenance.json) · [Migration map](migration.csv).
[Nt=1 analysis inventory](nt1_analysis_manifest.json): matched-beta diagnostics; folder c is the source Nt=2 reference target, and target_c is null.
[Matched-y simulation inputs](scaling_manifest.json) and [A100 execution plan](scaling_execution.json):
four approved ensembles at beta/Nt=1.5 and 1.55, sixteen 8,000-sweep chains.
Their c=0.5 folder identifies the source coarse-resolution grid; target_c is
null and these runs are excluded from fixed-c continuum inputs.
The manifest records the checksum-verified qbig recovery archive. Historical stage directories are retired after product verification.

- [c_0p5](c_0p5/)
- [c_1](c_1/)
- [c_2](c_2/)

- [migration.csv](migration.csv)
