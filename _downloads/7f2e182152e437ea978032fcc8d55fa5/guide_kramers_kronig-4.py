from pyimpspec import (
  generate_mock_data,
  mpl,
)
from pyimpspec.analysis.kramers_kronig import (
  evaluate_log_F_ext,
  suggest_num_RC,
)

data = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
tests = evaluate_log_F_ext(data)[0][1]
suggestion = suggest_num_RC(tests)

figure, axes = mpl.plot_kramers_kronig_tests(
  tests,
  suggestion,
  data,
  legend=False,
  colored_axes=True,
)
figure.tight_layout()