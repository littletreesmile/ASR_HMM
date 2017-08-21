from __future__ import division
from data_loader import parse_data
import numpy as np


FS = 22050  # sampling rate in [Hz]
TEN_MS = 10 / 1000  # 10 milliseconds


def set_length_10ms():
    """ Set number of samples for 10ms and ensure that is is even

    Returns:
        number of samples for 10ms
    """
    tmp_len = int(TEN_MS * FS)
    if tmp_len%2 != 0:
        tmp_len = tmp_len - 1
    return tmp_len

if __name__ == '__main__':

    file_name = 'IN_D1_FWC_S5_20170810_111645.snd'
    fs = 22050  # sampling rate
    frame_length_10ms = set_length_10ms()  # frame length in samples
    overlap_length = int(frame_length_10ms / 2)

    sound_data = parse_data(file_name)
    signal_channel_a = sound_data[294000:330000, 0]
    # signal_channel_b = sound_data[294000:330000, 1]
    subsignal_len = len(signal_channel_a)
    signal_channel_b = list(range(subsignal_len))
    num_pointing_5ms = int(subsignal_len / overlap_length)  # number of pointing by every 5ms
    current_frame = np.array([None] * frame_length_10ms)


    check = []
    for i in range(num_pointing_5ms):
        if i == num_pointing_5ms - 1:
            pass
        else:
            current_frame[:overlap_length] = current_frame[overlap_length:]
            current_frame[overlap_length:] = signal_channel_b[i*overlap_length:(i+1)*overlap_length]
            check.extend(current_frame)
    print (check)