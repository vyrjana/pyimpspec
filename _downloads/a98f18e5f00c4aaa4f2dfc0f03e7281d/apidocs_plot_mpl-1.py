import pyimpspec
from pyimpspec import mpl
from numpy import logspace, log10 as log

circuit = pyimpspec.generate_mock_circuits("CIRCUIT_1")[0]
data = pyimpspec.generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
f = data.get_frequencies()
figure, axes = mpl.plot_circuit(circuit, frequencies=f, label="TC-1", title="", legend=False, colored_axes=True)
figure.tight_layout()
mpl.show()

figure, axes = mpl.plot_nyquist(data, line=True)
figure.tight_layout()
mpl.show()

figure, axes = mpl.plot_bode(data, line=True)
figure.tight_layout()
mpl.show()