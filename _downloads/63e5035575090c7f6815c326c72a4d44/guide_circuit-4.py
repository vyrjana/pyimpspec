from pyimpspec import Circuit, DataSet, parse_cdc, simulate_spectrum
from pyimpspec import mpl
from numpy import logspace
circuit: Circuit = parse_cdc("R{R=20f:sol}(C{C=25e-6//1e-3:dl}[R{R=100/50/100:ct}W{Y=2.357e-3/inf/150%:diff}])")
data: DataSet = simulate_spectrum(circuit, frequencies=logspace(3, 0, num=16), label="Randles")
figure, axes = mpl.plot_nyquist(data)
figure.tight_layout()