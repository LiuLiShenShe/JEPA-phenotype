"""Dependency-free report table formatting."""

import numbers

import pandas as pd


def markdown_table(frame, digits=5):
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)
