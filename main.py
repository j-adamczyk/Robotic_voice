from modulator import module_audio
import numpy as np
from parameters import CHANNELS, CHUNK, NP_TYPE, PA_TYPE, RATE
import pyaudio as pa
import scipy.signal as signal
import time

p = pa.PyAudio()


def filter_audio(data):
    """
    Performs digital filter audio filtering.
    :param data: input sound data, numpy array
    :return: filtered sound data, numpy array
    """
    # elliptic filter with those parameters proved
    # experimentally to be the best
    b, a = signal.iirdesign(0.2, 0.3, 4, 60, ftype="ellip")
    data = signal.filtfilt(b, a, data).astype(NP_TYPE)
    return data


# function for manipulating input data
def callback(data, frame_count, time_info, flag):
    """
    Applies sound effects to input and outputs it in real time.
    :param data: input sound data, byte stream
    :param frame_count: unused, see PyAudio documentation
    :param time_info: unused, see PyAudio documentation
    :param flag: unused, see PyAudio documentation
    :return: tuple of: audio data after applying sound effects (numpy array),
    PyAudio constant to continue recording
    """
    data = np.frombuffer(data, dtype=NP_TYPE)
    data = filter_audio(data)
    data[abs(data) < 10] = 0

    data = module_audio(data, RATE, NP_TYPE)

    data[abs(data) < 10] = 0

    return data, pa.paContinue


def playback(callback_function):
    """
    Records audio in endless loop and plays it back with callback function.
    """
    stream = p.open(format=PA_TYPE,
                    rate=RATE,
                    channels=CHANNELS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback_function)

    stream.start_stream()

    # main input loop
    while True:
        time.sleep(1)

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    playback(callback)
