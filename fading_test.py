import numpy as np
import matplotlib.pyplot as plt

def simulate_fading(fading_type="rayleigh", K_dB=6, n_samples=10000):
    if fading_type == "rayleigh":
        fading = (np.random.normal(0, 1, n_samples) + 1j * np.random.normal(0, 1, n_samples)) / np.sqrt(2)
    elif fading_type == "rician":
        K = 10 ** (K_dB / 10)
        s = np.sqrt(K / (K + 1))
        sigma = np.sqrt(1 / (2 * (K + 1)))
        fading = (s + np.random.normal(0, sigma, n_samples) + 1j * np.random.normal(0, sigma, n_samples))
    else:
        raise ValueError("Unknown fading type")

    power_gain_dB = 10 * np.log10(np.abs(fading) ** 2)
    return power_gain_dB

rayleigh_dB = simulate_fading("rayleigh")
rician_dB = simulate_fading("rician", K_dB=15)

plt.hist(rayleigh_dB, bins=100, alpha=0.6, label='Rayleigh')
plt.hist(rician_dB, bins=100, alpha=0.6, label='Rician (K=6 dB)')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("Fading Gain (dB)")
plt.ylabel("Count")
plt.title("Fading Gain Distribution")
plt.legend()
plt.grid(True)
plt.show()