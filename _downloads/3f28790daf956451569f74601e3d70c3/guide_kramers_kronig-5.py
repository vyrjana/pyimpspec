from pyimpspec import mpl, generate_mock_data

data = generate_mock_data("CIRCUIT_8", noise=5e-2, seed=42)[0]
figure, axes = mpl.plot_nyquist(data)
figure.tight_layout()