import numpy as np
from parameters import CHANNELS, CHUNK, RATE


# ring modulator diodes constants, must be below 0
# 0.2 and 0.4 are paper's default values
VB = 0.2
VL = 0.4

# distortion control parameter, default value 4
H = 4

# number of samples for diode simulator; default value 1024
DIODE_SAMPLES = 1024

# modulating frequency MOD_F (in Hz)
MOD_F = 50

# sine wave for modulation
# global constance, since copying it is more efficient than calculating
wave = np.sin(2 * np.pi * np.arange(CHANNELS * CHUNK) * MOD_F / RATE) * 0.5


def diode_lookup(n_samples):
    """
    Creates a lookup table for simulating diodes.
    :param n_samples: number of samples in lookup table
    :return: lookup table, numpy array
    """
    result = np.zeros((n_samples,))

    for i in range(0, n_samples):
        v = abs(float(i - float(n_samples) / 2) / (n_samples / 2))
        if VB < v <= VL:
            result[i] = H * ((v - VB) ** 2) / (2 * VL - 2 * VB)
        elif v > VL:
            result[i] = H * v - H * VL + \
                        (H * (VL - VB) ** 2) / (2 * VL - 2 * VB)

    return result


class Waveshaper:
    """
    Merges two signal waves together.
    """
    def __init__(self, curve):
        self.curve = curve
        self.n_bins = self.curve.shape[0]

    def transform(self, samples):
        # normalize to 0 < samples < 2
        max_val = np.max(np.abs(samples))
        if max_val >= 1.0:
            result = samples / max_val + 1.0
        else:
            result = samples + 1.0
        result = result * (self.n_bins - 1) / 2
        return self.curve[result.astype(np.int)]


# diode simulator
diode = Waveshaper(diode_lookup(DIODE_SAMPLES))


def module_audio(data, rate, data_type):
    """
    np.sin(2 * np.pi * np.arange(data.shape[0]) * MOD_F / rate) * 0.5
    Simulates ring modulator.
    :param data: audio input data, numpy array
    :param rate: sampling rate, int
    :param data_type: numpy data type, e. g. int32
    :return: audio output after modulation, numpy array
    (same data type as input)
    """
    # scale to [-1, 1]
    scale = np.max(np.abs(data))
    if scale != 0:
        data = data.astype(np.float) / scale

    global wave

    tone = wave.copy()

    # junctions
    tone2 = tone.copy()  # to top path
    data2 = data.copy()  # to bottom path

    # invert tone, sum paths
    tone = -tone + data2  # bottom path
    data = data + tone2   # top path

    # top
    data = diode.transform(data) + diode.transform(-data)

    # bottom
    tone = diode.transform(tone) + diode.transform(-tone)

    result = data - tone

    # scale to [-1, 1]
    new_scale = np.max(np.abs(result))
    if new_scale != 0:
        result /= new_scale

    # scale back to values of input file
    result *= scale
    return result.astype(data_type)
