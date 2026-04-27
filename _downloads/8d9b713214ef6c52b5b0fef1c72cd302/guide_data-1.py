from pyimpspec import mpl
from pyimpspec import generate_mock_data

data = generate_mock_data("CIRCUIT_1")[0]
figure, axes = mpl.plot_nyquist(data)