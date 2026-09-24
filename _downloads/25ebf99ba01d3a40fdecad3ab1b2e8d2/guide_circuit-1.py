from pyimpspec import Circuit, parse_cdc
circuit: Circuit = parse_cdc("R(C[RW])")
drawing = circuit.to_drawing()
drawing.draw()