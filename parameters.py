import numpy as np
import pyaudio as pa

# those two constants have to be of analogous types
PA_TYPE = pa.paInt32
NP_TYPE = np.int32

# sound input parameters
CHANNELS = 2
CHUNK = 1024
RATE = int(pa.PyAudio().get_default_input_device_info()['defaultSampleRate'])
