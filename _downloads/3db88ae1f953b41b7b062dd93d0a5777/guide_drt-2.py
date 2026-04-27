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

data = generate_mock_data(cdc, noise=0)[0]
drt = calculate_drt_lm(data)

figure, axes = mpl.plot_nyquist(data, colors=dict(impedance="black"), markers=dict(impedance="o"))
mpl.plot_nyquist(drt, colors=dict(impedance="red"), markers=dict(impedance="+"), figure=figure, axes=axes)
figure.tight_layout()

figure, axes = mpl.plot_gamma(drt)
figure.tight_layout()