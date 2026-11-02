from pyimpspec import Circuit, parse_cdc
circuit: Circuit = parse_cdc("R(RC)(RW)")
drawing = circuit.to_drawing(running=True)
drawing.draw()