from pyimpspec import parse_cdc
circuit = parse_cdc("R{R=20f:sol}(C{C=25e-6//1e-3:dl}[R{R=100/50/100:ct}W{Y=2.357e-3/inf/150%:diff}])")
drawing = circuit.to_drawing()
drawing.draw()