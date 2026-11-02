#!/usr/bin/env python
from pyimpspec import (
    generate_mock_data,
)
from pyimpspec.analysis.drt import calculate_drt_lm
from pyimpspec import mpl

cdc = "R{R=140}(R{R=230}C{C=1e-6})(R{R=576}C{C=1e-4})(R{R=150}L{L=4e1})"
data = generate_mock_data(cdc, noise=5e-2, seed=42)[0]
drt = calculate_drt_lm(data)

figure, axes = mpl.plot_gamma(drt)
figure.tight_layout()

text_output = f"""
{drt.to_peaks_dataframe().to_markdown(index=False)}
""".strip()

with open("example_drt_lm_analysis_d.txt", "w") as fp:
    fp.write(text_output)
