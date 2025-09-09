# mic-vis

A Python package for microbeam visualization and analysis, specifically designed for processing and visualizing X-ray fluorescence (XRF) data from microbeam experiments.

## Installation

```bash
# Clone the repository
git clone git@github.com:grace227/mic-vis.git
cd mic-vis

# Install in development mode
pip install -e .
```

## Package Structure

```
mic_vis/
├── __init__.py          # Package initialization
├── common/              # Common utilities
│   ├── __init__.py
│   ├── readMDA.py      # MDA file reader
│   └── plot.py         # Plotting functions
├── bnp/                 # BNP-specific functionality
│   ├── __init__.py
│   └── io.py           # HDF5 I/O operations
└── s2idd/               # S2IDD tools
    └── __init__.py
```
