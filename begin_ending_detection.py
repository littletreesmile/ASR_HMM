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


def short_time_energy(sig):
    """ Calculate short time energy

    Args:
        sig: signals

    Returns:
        averge energy
    """
    abs_values = [abs(x) for x in sig]
    return sum(abs_values)


def zero_crossing_rate(sig):
    """ Calculate zero crossing rate

    Args:
        sig: signals

    Returns:
        zero crossing rate
    """
    zero_crossing = []
    for i in range(len(sig)-1):
        tmp = int(abs((sig[i+1] - sig[i])) / 2)
        zero_crossing.append(tmp)
    return sum(zero_crossing)


if __name__ == '__main__':

    file_name = 'IN_D1_FWC_S5_20170810_111645.snd'
    fs = 22050  # sampling rate
    frame_length_10ms = set_length_10ms()  # frame length in samples
    overlap_length = int(frame_length_10ms / 2)  # overlap length in samples

    sound_data = parse_data(file_name)
    signal_channel_a = sound_data[294000:330000, 0]
    # signal_channel_b = sound_data[294000:330000, 1]
    subsignal_len = len(signal_channel_a)
    signal_channel_b = list(range(subsignal_len))

    num_frames = int(subsignal_len / overlap_length) - 1  # number of frames by every 5ms
    current_frame = np.array([None] * frame_length_10ms)


    check = []
    STE = []  # list of average energy
    raw_start_found = False
    for i in range(num_frames):
        current_frame = signal_channel_b[i*overlap_length:i*overlap_length+frame_length_10ms]  # extract 10ms data
        ste = short_time_energy(current_frame)  # short time energy of the current frame
        zcr = zero_crossing_rate(current_frame)  #  zero crossing rate of the current frame

        if (ste > STE_UPPER) and (not raw_start_found):
            start_candidate = np.where(np.array(STE) < STE_LOWER)[-1]
            if start_candidate:
                raw_start = start_candidate[0]
            else:
                raw_start = i
            raw_start_found = True
        elif (ste < STE_LOWER) and raw_start_found:
            raw_end = i





        STE.append(ste)


        current_frame[:overlap_length] = current_frame[overlap_length:]
        current_frame[overlap_length:] = signal_channel_b[i*overlap_length:(i+1)*overlap_length]
        check.extend(current_frame)
    print (check)