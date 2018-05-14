"""Summary.

#CADL
Copyright Parag K. Mital 2016
"""
import numpy as np
from scipy.signal import hann


def ztoc(re, im):
    return np.sqrt(re**2 + im**2), np.angle(re + im * 1j)


def ctoz(mag, phs):
    return mag * np.cos(phs), mag * np.sin(phs)


def dft_np(signal, hop_size=256, fft_size=512):
    """
    Splits the input signal into multiple frames (len(signal)/hop_size) of length fft_size
    and applies the Discrete Fourier Transform to each of them. Returns the Real and Imaginary
    components of the frequency domain for each frame.
    :param signal: the signal to be processed
    :param hop_size: defines the number of frames to extract from the input signal (if
    hop_size is smaller than fft_size then most input points will belong to multiple frames.
    If it is greater than fft_size, some points will not belong to any frame)
    :param fft_size: the size of each frame to apply the DFT
    :return: the real matrix contains stacked rows, each corresponding to the ReX(k) values
    of an input frame (k from 0 to N/2). Row j corresponds to frame j. Similarly, the
    imaginary matrix contains the ImX(k) values.
    """
    # Calculate the number of frames that we'll extract from the audio file
    n_hops = len(signal) // hop_size
    s = []
    hann_window = hann(fft_size)
    # Divide the signal into n_hops frames of length fft_size
    for hop_i in range(n_hops):
        frame = signal[(hop_i * hop_size):(hop_i * hop_size + fft_size)]
        # Pad the last frame with 0s to make sure its length is equal to fft_size
        frame = np.pad(frame, (0, fft_size - len(frame)), 'constant')
        # Multiply the frame by the Hann window
        frame *= hann_window
        s.append(frame)
    s = np.array(s)
    # N is fft_size (the length of a frame)
    N = s.shape[-1]
    # Create a vector containing all the values of k from 0 to N/2
    k = np.reshape(np.linspace(0.0, (N // 2), N // 2), [1, N // 2])
    # All the values of t (from 0 to N-1)
    t = np.reshape(np.linspace(0.0, N - 1, N), [N, 1])
    # All the 2.pi.k.t/N for all values of k and all values of t (matrix)
    frequencies = np.dot(t, k * 2 * np.pi / N)
    # Line number j of this matrix contains the ReX(k) values of the j-th frame of s
    real = np.dot(s, np.cos(frequencies)) * (2.0 / N)
    imaginary = np.dot(s, np.sin(frequencies)) * (2.0 / N)
    return real, imaginary


def idft_np(re, im, hop_size=256, fft_size=512):
    N = re.shape[1] * 2
    k = np.reshape(np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [N // 2, 1])
    x = np.reshape(np.linspace(0.0, N - 1, N), [1, N])
    freqs = np.dot(k, x)
    signal = np.zeros((re.shape[0] * hop_size + fft_size,))
    recon = np.dot(re, np.cos(freqs)) + np.dot(im, np.sin(freqs))
    for hop_i, frame in enumerate(recon):
        signal[(hop_i * hop_size): (hop_i * hop_size + fft_size)] += frame
    return signal
