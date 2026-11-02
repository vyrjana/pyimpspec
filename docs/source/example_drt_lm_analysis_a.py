#!/usr/bin/env python
from pyimpspec import (
    generate_mock_data,
    parse_cdc,
)
from pyimpspec.analysis.drt import calculate_drt_lm
from pyimpspec import mpl

cdc = "R{R=140}(R{R=230}C{C=1e-6})(R{R=576}C{C=1e-4})(R{R=150}L{L=4e1})"
circuit = parse_cdc(cdc)
drawing = circuit.to_drawing()
drawing.draw()
