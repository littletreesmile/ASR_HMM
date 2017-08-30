from __future__ import division
from data_loader import parse_data
import matplotlib.pyplot as plt
import numpy as np


FS = 22050  # sampling rate in [Hz]
TEN_MS = 10 / 1000  # 10 milliseconds
IF = 55  # 55 zero crossings per 10 ms (one frame)


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
    signal_channel_b = sound_data[294000:330000, 1]
    subsignal_len = len(signal_channel_b)
    # signal_channel_b = list(range(subsignal_len))

    num_frames = int(subsignal_len / overlap_length) - 1  # number of frames by every 5ms
    current_frame = np.array([None] * frame_length_10ms)


    check = []
    STE = []  # list of average energy
    raw_start_found = False

    ste_list = []
    zcr_list = []
    for i in range(20):  # 20 5ms is 100ms
        current_frame = signal_channel_b[i * overlap_length:i * overlap_length + frame_length_10ms]  # extract 10ms data
        ste_list.append(short_time_energy(current_frame))  # short time energy of the current frame
        zcr_list.append(zero_crossing_rate(current_frame))  # zero crossing rate of the current frame

    zcr_list_mean = np.mean(np.array(zcr_list))
    zcr_list_std = np.std(np.array(zcr_list))
    ste_list_mean = np.mean(np.array(ste_list))
    ste_list_std = np.std(np.array(ste_list))

    IZCT = np.min(IF, zcr_list_mean + 2*zcr_list_std)
    MINSTE = min(0.25, ste_list_mean + ste_list_std)
    # STE_UPPER = 32 * MINSTE
    # STE_LOWER = 8 * MINSTE
    STE_UPPER = 4
    STE_LOWER = 0.5

    raw_end = []
    for i in range(num_frames):
        current_frame = signal_channel_b[i*overlap_length:i*overlap_length+frame_length_10ms]  # extract 10ms data
        ste = short_time_energy(current_frame)  # short time energy of the current frame
        zcr = zero_crossing_rate(current_frame)  #  zero crossing rate of the current frame

        if (ste > STE_UPPER) and (not raw_start_found):
            start_candidate = np.where(np.array(STE) < STE_LOWER)[0]
            if start_candidate.size:
                raw_start = start_candidate[-1]
            else:
                raw_start = i
            raw_start_found = True
        elif (ste < STE_LOWER) and raw_start_found:
            raw_end.append(i)
            # break

        STE.append(ste)

        current_frame[:overlap_length] = current_frame[overlap_length:]
        current_frame[overlap_length:] = signal_channel_b[i*overlap_length:(i+1)*overlap_length]
        check.extend(current_frame)

    t1 = np.array(range(subsignal_len)) /fs
    t2 = np.array(range(num_frames)) * overlap_length / fs
    plt.figure(2)
    plt.subplot(211)
    plt.plot(t1, signal_channel_b)
    plt.plot([t1[raw_start * overlap_length]]*2,[-0.2, 0.2], 'r')
    plt.plot([t1[raw_end[0] * overlap_length]]*2, [-0.2, 0.2], 'g')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.subplot(212)
    plt.plot(t2, STE)
    plt.plot([t2[raw_start]]*2, [-10, 10], 'r')
    plt.plot([t2[raw_end[0]]]*2, [-10, 10], 'g')
    plt.xlabel('Time [s]')
    plt.grid()
    plt.show()
