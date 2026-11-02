from pyimpspec import Circuit, parse_cdc
from pyimpspec import mpl
from numpy import logspace
circuit: Circuit = parse_cdc("R{R=20f:sol}(C{C=25e-6//1e-3:dl}[R{R=100/50/100:ct}W{Y=2.357e-3/inf/150%:diff}])")
figure, axes = mpl.plot_circuit(circuit, frequencies=logspace(3, 0, num=16), label="Randles", title="")
figure.tight_layout()