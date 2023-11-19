## About

This is a small project to test the features of [PyO3](https://pyo3.rs/main/), the Python/Rust binding.

## How to use

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [Python](https://www.python.org/downloads/)

### Installation

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

3. Build the Rust library and install it in the virtual environment:

```bash
maturin develop --release
```

4. Start the Jupyter lab:

```bash
python -m jupyter lab
```

Open mandelbrot.ipynb and run the cells.
