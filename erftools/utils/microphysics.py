import numpy as np

def p_sat(temp):
    tC = temp - 273.15  # Convert temperature from Kelvin to Celsius

    # Create masks for conditions
    mask_positive = tC > 0.0
    mask_negative = ~mask_positive

    # Initialize ps with zeros (same shape as temp)
    ps = np.zeros_like(temp)

    # Compute ps for tC > 0
    ps[mask_positive] = 6.112 * np.exp(17.62 * tC[mask_positive] / (tC[mask_positive] + 243.12))

    # Compute ps for tC <= 0
    ps[mask_negative] = 6.112 * np.exp(22.46 * tC[mask_negative] / (tC[mask_negative] + 272.62))

    return ps

