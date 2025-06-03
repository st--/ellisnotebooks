# Marimo notebooks for ELLIS Manchester summer school 2025

You can run the notebooks either locally (faster, recommended) or online (slower but no installation required).

## Run locally

First, [install marimo](https://docs.marimo.io/getting_started/installation/):
```bash
pip install marimo
```
(See marimo docs for instructions using `uv` or `conda` instead of `pip`).

Check that it works by running `marimo tutorial intro`.

Clone this repository on your own computer:
```bash
git clone https://github.com/st--/ellisnotebooks.git
```

You can edit (or create new) marimo notebooks using
```bash
marimo edit --sandbox <notebook>.py
```

## Run online

Go to [marimo.new](https://marimo.new/), click on "New", then on "Open from URL...", and paste the URL from GitHub, e.g. for the `0_marimo.py` notebook, right-click and copy the link which is `https://github.com/st--/ellisnotebooks/blob/main/0_marimo.py`, and paste it into the Import .py Notebook URL field.

I encountered some issues with imports not running properly. I could fix this by adding a new cell, copying the relevant import statements in there again, and running this new cell.
