import pyimpspec
from pyimpspec import mpl
import pyimpspec.plot.colors as colors

data = pyimpspec.generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
evaluations = pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(data)

figure, axes = mpl.plot_log_F_ext(
  evaluations,
  projection="3d",
)
figure.tight_layout()
mpl.show()

figure, axes = mpl.plot_log_F_ext(
  evaluations,
  projection="2d",
)
figure.tight_layout()
mpl.show()