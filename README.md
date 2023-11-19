## About

This is a small project demonstrating using Rust to speedup Python. See my blog post
[part 1](https://aspcompiler.github.io/posts/how-rust-x-speedup-over-python-1/) and
[part 2](https://aspcompiler.github.io/posts/how-rust-x-speedup-over-python-2/).

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
