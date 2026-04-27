from pyimpspec import (
  generate_mock_data,
  mpl,
)
from pyimpspec.analysis.kramers_kronig import (
  evaluate_log_F_ext,
  suggest_num_RC,
)

data = generate_mock_data("CIRCUIT_4", noise=5e-2, seed=42)[0]
evaluations = evaluate_log_F_ext(data, min_log_F_ext=-1.0, max_log_F_ext=1.0, num_F_ext_evaluations=20)
figure, axes = mpl.plot_log_F_ext(evaluations)
figure.tight_layout()

figure, axes = mpl.plot_log_F_ext(evaluations, projection="2d", legend=False)
figure.tight_layout()

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