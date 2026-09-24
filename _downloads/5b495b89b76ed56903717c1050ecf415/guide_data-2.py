from pyimpspec import DataSet
from pyimpspec import mpl
from pyimpspec import generate_mock_data

data = generate_mock_data("CIRCUIT_1")[0]
figure, axes = mpl.plot_bode(data)

data.low_pass(1e3)
data.high_pass(1e1)
figure, axes = mpl.plot_bode(data)