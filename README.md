# AIedes Albopictus Datasets

Code and pipelines for constructing the harmonised European surveillance–climate datasets for *Aedes albopictus* described in the accompanying paper:

> **Harmonized European surveillance–climate datasets for *Aedes albopictus***
> Biazzo I., Orfei L., Schuh L., Consoli S., Stilianakis N. I., Markov P. V.
> European Commission, Joint Research Centre (JRC), Ispra, Italy

## Purpose

Quantitative modelling of mosquito vector distribution and abundance is central to surveillance, risk assessment, and public health decision-making. However, systematic comparison of modelling approaches is hindered by the lack of harmonised datasets that consistently link surveillance observations with climate covariates in a standardised and reusable format.

This repository provides fully scripted, reproducible pipelines that produce two complementary, analysis-ready datasets pairing *Aedes albopictus* surveillance data with gridded climate variables across Europe. Both datasets rely exclusively on open data sources and are designed to support benchmarking, methodological comparison, and reuse across ecological and epidemiological studies.

## Datasets

| # | Dataset | Surveillance source | Climate source | Resolution | Output location |
|---|---------|-------------------|----------------|------------|-----------------|
| 1 | **ECDC polygon-to-grid** | ECDC mosquito maps (NUTS-3 polygons) | CORDEX (~12 km) or ERA5-Land (~9 km) | Grid point × decadal climatology | `data/classifier/data/outputs/` |
| 2 | **AIMSurv trap-level** | AIMSurv v2.3 (Darwin Core Archive) | ERA5-Land (~9 km) | Trap × daily / monthly time series | `data/counter/output_data/` |

**Dataset 1** spatially joins polygon-level presence/absence labels to individual climate grid points, producing a dataset suitable for species distribution modelling and binary classification.

**Dataset 2** pairs individual trap observations (positive detections and confirmed zero-counts) with 89-day daily climate histories and 3-month summaries, suitable for time-resolved abundance modelling.

## Repository Structure

```
AIedes_data/
├── README.md                          # ← this file
├── paper/
│   ├── main.tex                       # Manuscript source
│   └── references.bib                 # Bibliography
└── data/
    ├── classifier/                    # Dataset 1 pipeline
    │   ├── README.md                  # Full documentation
    │   ├── pair_ecdc_copernicus_data.py
    │   ├── make_classifier_database.sh
    │   ├── src/                       # Processing modules
    │   └── data/
    │       ├── inputs/                # ECDC GDB + climate NetCDFs
    │       └── outputs/               # ⬅ Generated Dataset 1
    └── counter/                       # Dataset 2 pipeline
        ├── README.md                  # Full documentation
        ├── make_counter_dataset.sh
        ├── src/                       # Processing modules
        ├── input_data/                # AIMSurv DwC-A + climate NetCDFs
        ├── output_data/               # ⬅ Generated Dataset 2
        └── output_stats/              # QA summary + plots
```

## Quick Start

### Dataset 1 — ECDC polygon-to-grid

```bash
cd data/classifier
python pair_ecdc_copernicus_data.py --year 2020 --climate-source cordex
```

See [`data/classifier/README.md`](data/classifier/README.md) for full documentation.

### Dataset 2 — AIMSurv trap-level

```bash
cd data/counter
bash make_counter_dataset.sh
```

See [`data/counter/README.md`](data/counter/README.md) for full documentation.

## Installation

### Method 1: Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/AIedes_data.git
cd AIedes_data

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Using conda

```bash
# Clone the repository
git clone https://github.com/your-org/AIedes_data.git
cd AIedes_data

# Create and activate environment
conda env create -f environment.yml
conda activate aiedes-data
```

### Method 3: Using existing climate_env (recommended for maintainers)

If you already have the `climate_env` environment set up:

```bash
# Clone the repository
git clone https://github.com/your-org/AIedes_data.git
cd AIedes_data

# Activate existing environment
conda activate climate_env
# All dependencies are already satisfied!
```

### Development Setup

For development work (includes testing and code quality tools):

```bash
pip install -r requirements-dev.txt
```

### Prerequisites

- **Python**: ≥3.12 (tested with 3.12.9 in climate_env)
- **Copernicus CDS API**: Climate downloads require API credentials — see the individual READMEs for setup instructions.

## Data Sources

| Source | Description | License | Link |
|--------|-------------|---------|------|
| **ECDC** | Polygon-level mosquito surveillance (NUTS-3) | ECDC data use terms | [ecdc.europa.eu](https://www.ecdc.europa.eu/en/disease-vectors/surveillance-and-disease-data/mosquito-maps) |
| **AIMSurv v2.3** | Trap-level mosquito surveillance (Darwin Core Archive) | CC0 1.0 | [GBIF IPT](https://ipt.gbif.es/resource?r=aimsurv) · [Zenodo](https://doi.org/10.5281/zenodo.10985325) |
| **ERA5-Land** | Climate reanalysis at 0.1° (~9 km) | Copernicus License | [CDS](https://doi.org/10.24381/cds.e2161bac) |
| **CORDEX** | Regional climate projections at 0.11° (~12 km) | Modelling centre terms | [cordex.org](https://cordex.org/) |

## Citation

If you use these datasets or pipelines, please cite:

> [Authors] (2025). Harmonized European surveillance–climate datasets for *Aedes albopictus*. [Journal/Repository]. [DOI]

See the individual dataset READMEs for source-specific citations.

## License

- **Pipeline code**: [Add your license]
- **AIMSurv data**: CC0 1.0 Public Domain Dedication
- **ECDC data**: [ECDC legal notice](https://www.ecdc.europa.eu/en/legal-notice)
- **ERA5-Land data**: Copernicus License
- **CORDEX data**: Subject to data use terms of respective modelling centres
