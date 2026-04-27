from pyimpspec import mpl
from pyimpspec import generate_mock_data

data = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]
figure, axes = mpl.plot_nyquist(data, colors={"impedance": "black"})
figure.tight_layout()