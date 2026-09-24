from pyimpspec import Circuit, parse_cdc
circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
drawing = circuit.to_drawing()
drawing.draw()