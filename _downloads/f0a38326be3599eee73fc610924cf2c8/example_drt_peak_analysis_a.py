#!/usr/bin/env python
from pyimpspec import (
    generate_mock_data,
    parse_cdc,
)
from pyimpspec.analysis.drt import calculate_drt_tr_rbf
from pyimpspec import mpl

cdc = "R{R=100}(R{R=300}C{C=5e-6})(R{R=450}C{C=1e-5})"
circuit = parse_cdc(cdc)
drawing = circuit.to_drawing()
drawing.draw()
