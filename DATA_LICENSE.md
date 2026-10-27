# Data licence and attribution

The **code** in this repository is licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
The **data** is not ours. It is **not stored in this repository** (too large for GitHub); download it from the sources below into the paths shown. Original licences:

| Files | Origin | Licence |
|---|---|---|
| `data/raw/mcns/neurons.csv.gz`, `data/raw/mcns/connections_princeton.csv.gz` | MaleCNS v1.0 (Janelia FlyEM), exported from FlyWire Codex | CC BY 4.0 |
| `data/raw/banc/neurons.csv.gz`, `data/raw/banc/connections_princeton.csv.gz` | BANC v888, exported from FlyWire Codex | CC BY 4.0 |
| `data/external/banc_dataverse_v3/*` (incl. the optional NBLAST matrix) | BANC Harvard Dataverse, doi:10.7910/DVN/7WTH1N (version 3) | CC BY 4.0 |
| `data/splits/`, `maps/`, `results/`, and the processed data in the Kaggle package | derived by this project | CC BY 4.0 |

Under CC BY 4.0 you may share and adapt the data for any purpose, **provided you give appropriate credit**.
If you use it, cite the original datasets:

- **BANC:** Bates, A. S., Phelps, J. S., Kim, M., … Lee, W.-C. A. (2026). *Distributed control circuits across
  a brain-and-cord connectome.* Nature 656, 957–970. https://www.nature.com/articles/s41586-026-10735-w ·
  Data: https://doi.org/10.7910/DVN/7WTH1N
- **MaleCNS:** *Sexual dimorphism in the complete Drosophila male CNS connectome.* Cell (2026).
  https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6 · Data: https://male-cns.janelia.org/download/
- **FlyWire Codex** (export platform): https://codex.flywire.ai
- **FlyWire** whole-brain connectome and annotation (source of the shared cell-type names):
  Dorkenwald, S., et al. (2024). Nature 634, 124–138; Schlegel, P., et al. (2024). Nature 634, 139–152.

Download dates, file IDs and sizes: [`data/external/SOURCES.md`](data/external/SOURCES.md).
Codex data is live and changes. These files are the snapshots this project was built on.
