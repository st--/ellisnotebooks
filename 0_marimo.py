import marimo

__generated_with = "0.12.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Marimo quickstart

        Marimo is an interactive Python notebook, similar to Jupyter lab/notebooks, but with a few key differences:

        - Marimo is **reactive**: when you change the values of variables in one cell, all other cells that use that variable will be updated as well
            - After making changes in a cell, you still have to execute that cell to update its output (as well as all dependents); you can press `Ctrl+Enter` to execute & keep focus on the same cell (useful if you want to see the effects of the change and then make further changes in the same cell), `Shift+Enter` to execute & move to the next cell, or `Ctrl+Shift+Enter` to execute & move to the previous cell.
        - This means that you can only use 'global' variable names **once** in a cell per notebook. This also applies to functions!
            - Make variables and functions local to a cell by using names starting with an underscore `_`
            - Wrap code of a cell with local variables and function definitions inside another function, e.g. `def _(): ...` and then call that function at the end of the cell
        - Outputs of a cell are generally shown *above* its code, not below (one exception is matplotlib figures where you end a cell with `plt.show()`)
        - Useful keyboard shortcut: `Ctrl+/` toggles the comment status of the current line (or all the lines included in a selection)
        - Another gotcha: you can split cells (`Ctrl+Shift+'`) and undo cell splits (`Ctrl+Z`), but you cannot easily merge cells (you'd have to copy the code from one cell into the other and then delete the left-over empty cell)
        """
    )
    return


@app.cell
def _():
    "The return value of a cell will be displayed above the cell"
    return


@app.cell
def _():
    "The return value of a cell is the last line of code"
    None
    return


@app.cell
def _():
    a = 1
    # a=3  # press Ctrl+/ while your cursor is in this line (or while it is selected) to toggle the comment status
    return (a,)


@app.cell
def _(a, b, mo):
    c = 10**a + b
    mo.md(f"""Variable c has value {c}. This cell output will be updated as soon as one of its ancestor cells changes""")
    return (c,)


@app.cell
def _():
    b = 5  # gotcha: cells can be placed in any order -- only the actual variable dependencies matter
    return (b,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
