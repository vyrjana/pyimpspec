#!/usr/bin/env python
from pyimpspec import Circuit, parse_cdc

circuit: Circuit = parse_cdc("R{:ser}(R{:hf}C{:hf})(R{:mf}C{:mf})(R{:lf}C{:lf})")
drawing = circuit.to_drawing()
drawing.draw()
