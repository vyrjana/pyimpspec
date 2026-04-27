from pyimpspec import perform_zhit, generate_mock_data
from pyimpspec import mpl

valid = generate_mock_data("CIRCUIT_2", noise=5e-2, seed=42)[0]
invalid = generate_mock_data("CIRCUIT_2_INVALID", noise=5e-2, seed=42)[0]
zhit = perform_zhit(invalid)

figure, axes = mpl.plot_bode(
  valid,
  legend=False,
  colors={"magnitude": "black", "phase": "black"},
  markers={"magnitude": "o", "phase": "s"},
)
mpl.plot_bode(
  invalid,
  legend=False,
  colors={"magnitude": "black", "phase": "black"},
  markers={"magnitude": "x", "phase": "+"},
  figure=figure,
  axes=axes,
)
mpl.plot_bode(
  zhit,
  line=True,
  legend=False,
  figure=figure,
  axes=axes,
)

lines = []
labels = []
for ax in axes:
  li, la = ax.get_legend_handles_labels()
  lines.extend(li)
  labels.extend(la)

axes[1].legend(lines, labels, loc=(0.03, 0.13))
figure.tight_layout()