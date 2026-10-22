import numpy as np

def gauge_twirling_samplex_parameters(samplex, ntwirls):
    parameter_values_specs_names = [s.name for s in samplex.inputs().specs]
    try:
        parameter_values_spec_index = parameter_values_specs_names.index("parameter_values")
        bitarr_size = samplex.inputs().specs[parameter_values_spec_index].shape[0]
        random_bits = np.random.randint(2, size=(ntwirls, bitarr_size))
        return {
            "parameter_values": np.array(random_bits)
        }
    except ValueError:
        return {}