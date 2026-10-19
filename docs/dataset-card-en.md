# uniARPAT crystal-q1-v1 dataset card

[中文](dataset-card-zh.md) · [Sources, citations and license review](dataset-sources.md) · Updated: 2026-10-19

## Purpose and version

This is a **processed computational dataset** for joint prediction of electronic and phonon densities of states
(eDOS/phDOS) from crystal structures. Each record contains a Materials Project identifier, a complete crystal input,
two discretized spectra, coverage masks, and a valence-electron count. It supports model development on frozen splits.
The targets are calculated properties, not experimental measurements or results recomputed under one uniform protocol.

`crystal-q1-v1` fixes Q1 membership, E0/P0 grids, and the complete-structure repair convention.
Its data manifest SHA-256 is `8e1b0adff516835d52d9fb0321dae5e66fb9a2ac5fd964d317e1dd8647de4b80`.
The proposed release tag is `data-crystal-q1-v1`; nothing has been uploaded. The source notice records source-specific
terms and attribution requirements. This release grants no new uniform license over the archive.
All 23,306 materials are verified as `BY-C`, with no `BY-NC`, in MP's official 2026-04-13 snapshot.
The package supports the current ZP and local-update path. It excludes the legacy periodic table, mendeleev element-feature
snapshots, model weights, and training outputs.

## Sources and size

Structures and all eDOS originate from [Materials Project](https://docs.materialsproject.org/downloading-data/aws-opendata).
phDOS is selected from MP, [PhononDB / NIMS MDR](https://doi.org/10.48505/nims.4197), and
[JARVIS-DFT / NIST](https://pages.nist.gov/jarvis/databases/).
Counts below come from joining the frozen IDs to the historical processed table, not from upstream database totals.

| Split | Stored materials | Default view | MP phDOS | PhononDB phDOS | JARVIS phDOS |
|---|---:|---:|---:|---:|---:|
| train | 18,706 | 18,704 | 14,946 | 1,901 | 1,859 |
| valid | 2,313 | 2,313 | 1,863 | 246 | 204 |
| test | 2,287 | 2,287 | 1,835 | 243 | 209 |
| Total | 23,306 | 23,304 | 18,644 | 2,390 | 2,272 |

Source counts describe **stored records**. Two PhononDB records are excluded from the default train view,
which therefore uses 1,899 records from that source. Stored structures contain 275,049 atoms in total,
with at most 192 atoms per structure; the default train view has at most 160.
Acquisition and parent-pool construction took place in September 2026; complete structures were repaired on October 4.
This is a historical snapshot, not a live upstream mirror. Parent documentation identifies MP's 2026.04 series,
PhononDB PBEsol, and JARVIS optB88vdW among its sources; complete per-task computational settings are not included.

## Processing logic: sources to model inputs

`Acquisition and ID mapping → paired spectra → axis conversion and binning → spike handling → frozen splits → Q1 filtering → complete-structure repair → loader normalization`

1. **Material identity.** Historical MP identifiers and external-source mappings are translated to canonical MP IDs
   through MP summary records, with one primary record per material. Inputs use MP summary structures; eDOS uses an
   attributable MP DOS task. Identity/lineage checks do not establish agreement of every computational condition across structure and spectra.
2. **Spectrum selection.** eDOS uses MP exclusively. phDOS priority is MP `pheasy` → MP `dfpt` → PhononDB → JARVIS.
   PhononDB DOS was recomputed locally from published phonopy inputs with a `20×20×20` mesh and the tetrahedron method.
   An external phDOS source does not imply that the corresponding eDOS is external.
3. **Axes and amplitudes.** eDOS energies are referenced to the Fermi energy of **their own DOS task**; up/down channels
   are summed, retaining the available total when the down channel is absent. Phonon frequencies in THz are multiplied
   by `33.35641` to obtain cm⁻¹, while densities are divided by that factor, preserving the integral of this conversion.
   JARVIS phonon intensities are scaled by `3N / native full-spectrum area`, using the complete MP input primitive-cell atom count.
   MP/PhononDB retain their native DOS counting scale, without an additional uniform forced 3N normalization.
4. **Binning and coverage.** Finite source points enter left-inclusive, right-exclusive bins. With at least two points,
   the stored value is their trapezoidal integral divided by the full bin width; with one point it is that value;
   with no points it is zero with `mask=False`. Bin boundaries are not interpolated, so rebinning is not strictly integral conserving.
   Coverage describes source sampling, and an uncovered stored zero is not an observed zero density.
5. **Spike handling.** Per-bin p99.9 thresholds are estimated from covered values in the historical parent pool.
   Isolated above-threshold spikes with neighbors below half their height are clipped; wide peaks and shoulders are retained.
   These are **parent-pool-level** preprocessing thresholds, not thresholds fitted exclusively on train. This release preserves the results.
6. **Splitting and Q1 filtering.** The 24,988-record parent pool was grouped/stratified into approximately 8:1:1 splits;
   Q1 then excluded 1,682 records without reassigning the remaining splits or order. Of these, 1,668 had full-spectrum
   `C_full/N_val < 0.5`; another 14 had unreadable native spectra and abnormally low scale labels.
   `C_full` is not the E0-window total. `N_val` prioritizes occupied-DOS integration with cell-ratio auditing and falls back to element counts when necessary.
7. **Complete structures.** MP inputs use pymatgen standard primitive cells with `symprec=0.1 Å`.
   The old cache capped atoms at 80. This version completes 55 affected structures and corrects 54 composition-fallback
   valence counts, preserving spectra and row order. `mp-aaabidtd` and `mp-aaabilxy` fall below the existing quality threshold
   after correction; they remain stored but are filtered from default train.
8. **Training transformation.** Disk arrays contain processed DOS densities. Current SumNorm divides each spectrum by
   its discrete sum in the loader to form a dimensionless shape; an all-zero spectrum remains zero under the denominator epsilon.
   That sum is not directly a physical integral;
   window integrals also require bin widths. Spectral scale is modeled separately; blind/oracle do not denote separate released target arrays.

Historical methods are available in Git at `e24d2a4^:tools/getdata/a4_process.py`, `a4_pass2.py`, and `recompute_phonondb.py`.
Q1 and complete-structure steps are implemented in `tools/data/q1_rebuild.py` and `tools/data/migrate_crystal_q1.py`.

## Axes, units and schema

| Target | Axis | Grid | Meaning of stored amplitude |
|---|---|---|---|
| eDOS | `E−E_F`, eV | E0: [-6, 6], 128 bins, width 0.09375 eV | states/eV in the native MP DOS calculation cell after binning/clipping; not uniformly divided by atom count |
| phDOS | Frequency, cm⁻¹ | P0: [-280, 980], 64 bins, width 19.6875 cm⁻¹ | Converted mode density, including JARVIS's 3N scaling convention; the finite-window binned integral need not equal 3N |

Amplitudes are **densities, not bin integrals**, and must not be described as per-atom DOS.
Equality of native DOS cells and current input cells has not been certified for every record.
Negative phonon frequencies retain the upstream representation of imaginary modes; the dataset is not a dynamically stable-only collection.

Each `data/crystal_q1_v1/{split}/` has 11 arrays. Below, `s` denotes a split, `M` its material count, and `S` its total atom count.

| File pattern | Shape / dtype | Meaning |
|---|---|---|
| `cell_s.npy` | [M,3,3] / float32 | Lattice row vectors, Å |
| `frac_coords_s.npy` | [S,3] / float32 | Dimensionless fractional coordinates; Cartesian coordinates are `frac @ cell` |
| `atomic_numbers_s.npy` | [S] / uint8 | Atomic numbers |
| `atom_offsets_s.npy` | [M+1] / int64 | Structure i occupies `[offsets[i]:offsets[i+1]]` |
| `n_atoms_s.npy` | [M] / int32 | Complete atom count N |
| `nvalence_s.npy` | [M] / float32 | Audited/fallback valence count for scale supervision; not a strictly label-independent input feature |
| `s_index.npy` | [M] / Unicode | Canonical MP IDs defining row alignment |
| `edos_tgtdos_s.npy`, `phdos_tgtdos_s.npy` | [M,128], [M,64] / float32 | Processed density targets |
| `edos_mask_s.npy`, `phdos_mask_s.npy` | [M,128], [M,64] / bool | True indicates finite source samples in the bin; training configuration determines loss masking |

There is no on-disk atom padding or lattice sentinel. `manifest.json` records shapes, dtypes, hashes, repairs, and exclusions.
The archive's 8 `data/reference/` files contain frozen splits, grids, the valence-count fallback table, exclusions, and historical electron-count/truncation audits:
`split_v2.yaml`, `grids.json`, `c2b_grids_report.json`, `q1_excluded_records.json`,
`valence_by_element.json`, `edos_valence_by_material.parquet`, `edos_truncation_by_material.parquet`, and `edos-label-audit.md`.
Some reference tables cover all 24,988 parent materials: filter by current IDs, and use the current manifest/arrays for the 54 corrected counts.
Together with the 33 current cache arrays and their manifest, the archive contains 42 files, all byte-identical to the frozen local files.

## What the splits guarantee

The historical split groups **absolute atom-count compositions**, balancing band-gap classes, crystal systems, and spectral
difficulty, and records seed42. It is not an independent random material split. The original algorithm used the then-capped
80-atom compositions. A new audit of the complete stored structures finds no material-ID or complete absolute-composition overlap.
However, reduced compositions overlap: train–valid, train–test, and valid–test share **254, 253, and 44** reduced compositions.
These count intersecting composition keys, not duplicate materials. This is not a benchmark with entirely unseen reduced chemical formulas.

Use train/valid for model development and checkpoint selection. Historical test data have been used for diagnostics;
test is not an untouched final holdout. This release audit checked test files, provenance, compositions, and target equality only,
without running models or calculating performance metrics.

## Retrieval, restoration and provenance

Code belongs in Git; the data archive is prepared as a GitHub Release asset candidate outside ordinary Git history.
See `release-manifest.json` for its exact size and SHA-256.
The candidate folder contains `README-data.md` (Chinese), `README-data.en.md` (this card), `SOURCE_NOTICE.md`, the archive,
`material-provenance.csv.gz`, `provenance-audit.json`, `release-manifest.json`, `verification.json`, and `SHA256SUMS`.
Additional assets `mp-license-records.csv.gz` and `mp-license-audit.json` preserve MP license metadata.
The legacy element tables and their dedicated audit are outside this release.
The per-material CSV contains IDs, splits, row indices, default-view flags, spectrum sources/references, zero-window flags, historical structure references,
and historical `ph_weight`. **The current loader does not read this historical weight; it is not evidence of source-weighted training.**
MP phonon references retain method categories rather than recovered unique task IDs. Source URLs are constructed from identifiers,
and their long-term availability is not guaranteed.

Place all assets in a clean code checkout. An existing manifest and reference files in a fresh clone are expected.
The following preflight permits filling missing files or overwriting identical content, stopping on a different version:

```bash
sha256sum -c SHA256SUMS
python - <<'PY'
from pathlib import Path
import hashlib, json
for name, item in json.loads(Path('release-manifest.json').read_text())['files'].items():
    path = Path(name)
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], name
print('Existing payload files match; ready to extract.')
PY
tar -xzf uniARPAT-crystal-q1-v1.tar.gz
```

With project dependencies installed, validate arrays using `utils.crystal_cache.crystal_data_contract` and load through
`utils.dataset.Dos_Dataset`. Restoration has been verified without old checkpoints or original-server absolute paths.
The archive omits complete upstream archives and historical processing environments: restoring frozen files is reproducible;
reacquiring live upstream data is not guaranteed to reconstruct them byte for byte.
Regenerate provenance using `tools/data/export_release_provenance.py --cache ... --processed ... --out <fresh-directory>`.
It checks existing spectra/masks against the parent table and exports metadata without rebuilding targets or splits.

## Limitations, citation and maintenance

Functionals, magnetic states, calculation geometries, and detailed settings are not fully harmonized. In a historical valid
diagnostic subset of 394 materials, 125 input structures failed a strict match to the DOS calculation structures.
This is a subset statistic, not a dataset-wide error rate. Finite windows, coverage, and clipping affect spectral areas and details.
There are 7 all-zero phDOS window spectra in train, 1 in valid, and 0 in test; they match the parent table and remain in default views.
No eDOS spectrum is all-zero. These are not certified physical zero spectra; their causes have not been individually established.
Locate them using `phdos_zero_window` in the provenance CSV; changing the filtering requires a separate data-version decision.
Cite `crystal-q1-v1`, its manifest hash, actual train exclusions, grids, and normalization alongside MP, PhononDB, and JARVIS.
No independent dataset paper or DOI exists yet; do not invent authors or a DOI. Add the permanent release URL after publication.
Upstream references and licensing evidence are listed in the source notice.

Documentation organization follows [Matbench metadata](https://github.com/materialsproject/matbench/blob/main/matbench/matbench_v0.1_dataset_metadata.json),
[MPtrj version/citation notes](https://github.com/CederGroupHub/chgnet#mptrj-dataset), and
[Hugging Face Dataset Cards](https://huggingface.co/docs/hub/datasets-cards).
These are documentation references, not data sources for this archive. Report issues/corrections through the project repository;
changes to membership, targets, grids, or the complete-structure contract require a new data version.
