from pyimpspec import parse_cdc
circuit = parse_cdc("R(RC)(RC)CL")
elements = circuit.get_elements()
custom_labels = {
    elements[0]: r"$R_{\rm ser}$",
    elements[1]: r"$R_1$",
    elements[2]: r"$C_1$",
    elements[3]: r"$R_k$",
    elements[4]: r"$C_k$",
    elements[5]: r"$C_{\rm ser}$",
    elements[6]: r"$L_{\rm ser}$",
}
circuit.to_drawing(custom_labels=custom_labels).draw()