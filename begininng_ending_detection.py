from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from audio_activity_detection.data_loader import parse_data
from collections import namedtuple


FS = 22050  # sampling rate in [Hz]
TEN_MS = 10 / 1000  # 10 milliseconds
IF = 55  # 55 zero crossings per 10 ms (one frame)
THRESHOLD = namedtuple('THRESHOLD', 'IE, ste_mean, ste_std, MINSTE, T_lower, T_upper, IF, stzcr_mean, stzcr_std, T_zc')
MI5 = 5  # multiple of 5 frames, intotal 50 ms
MI10 = 10

class aad(object):
    """ Audio activity detection """
    def __init__(self, file_name, fs):
        self.file_name = file_name
        self.fs = fs
        self.frame_length_10ms = self.nSamples_ms(period=TEN_MS)  # frame length in samples
        self.ste_section = np.empty(self.frame_length_10ms) * np.nan
        # period of calculation of statistics for coarse search. 50 ms period (5 frames)
        self.stat_coarse_section = np.empty(self.frame_length_10ms * MI5) * np.nan
        self.stzcr_section = np.empty(self.frame_length_10ms * MI10) * np.nan  # no more than 10 frames
        self.pointer = 0
        self.signal_channel_2 = self.show_data()
        self.threshold_found = False  # has noise level (silence period) been known
        self.threshold = THRESHOLD
        self.coarse_Tl_found = False  # lower threshold in the coarse start search is found
        self.coarse_Tu_found = False  # upper threshold in the coarse start search is found
        self.coarse_start_candidate = None
        self.coarse_start = None
        self.fine_start = None
        self.coarse_end = None
        self.fine_end = None
        self.start_found = False
        self.coarse_end_found = False
        self.fine_end_found = False
        self.check_ste = []
        self.check_zcr = []

    def threshold_init(self):
        # thresholds for STE
        self.threshold.IE = 0.25
        self.threshold.ste_mean = np.mean(self.stat_coarse_section)  # mean of the signals in STE section
        self.threshold.ste_std = np.std(self.stat_coarse_section)  # standard deviation of the signals in STE section
        self.threshold.MINSTE = min(self.threshold.IE, self.threshold.ste_mean + self.threshold.ste_std)
        self.threshold.T_lower = 8 * self.threshold.MINSTE
        self.threshold.T_upper = 32 * self.threshold.MINSTE

        # thresholds for STZCR
        zcr_list = np.array(self.obtain_zcr(self.stat_coarse_section))
        self.threshold.IF = 0.25  # (number of zc) / (number of samples in one frame)
        self.threshold.stzcr_mean = np.mean(zcr_list)
        self.threshold.stzcr_std = np.std(zcr_list)
        self.threshold.T_zc = min(self.threshold.IF, self.threshold.stzcr_mean + self.threshold.stzcr_std)

        self.threshold_found = True

    def nSamples_ms(self, period=TEN_MS):
        """ Set number of samples for 10ms and ensure that the output is even.

        Args:
            period: the time period to be calculated [ms]
            fs: sampling rate [Hz]

        Returns:
            number of number of samples for 10ms

        """
        tmp_len = int(period * self.fs)
        if tmp_len % 2 != 0:
            tmp_len = tmp_len - 1
        return tmp_len

    def show_data(self):
        sound_data = parse_data(self.file_name)
        # return sound_data[30000:76000, 1]
        return sound_data[30000:300000, 1]

    def update_ste_section(self, overlap_length):
        self.ste_section[:overlap_length] = self.ste_section[
                                            overlap_length:]  # replace the 1st half section by the 2nd half
        self.ste_section[overlap_length:] = self.signal_channel_2[
                                            self.pointer:self.pointer + overlap_length]  # load 5ms data to the 1st half

    def update_stzcr_section(self, overlap_length):
        self.stzcr_section[:(self.frame_length_10ms * MI10 - overlap_length)] = self.stzcr_section[overlap_length:]
        self.stzcr_section[(self.frame_length_10ms * MI10 - overlap_length):] = \
            self.signal_channel_2[self.pointer:self.pointer + overlap_length]

    def update_stat_coarse_section(self, overlap_length):
        self.stat_coarse_section[:(self.frame_length_10ms * MI5 - overlap_length)] = self.stat_coarse_section[overlap_length:]
        self.stat_coarse_section[(self.frame_length_10ms * MI5 - overlap_length):] = \
            self.signal_channel_2[self.pointer:self.pointer + overlap_length]


    def update_sections(self):
        """ Update Short-term Energy (STE) section. """
        overlap_length = int(self.frame_length_10ms / 2)  # overlap length in samples
        self.update_ste_section(overlap_length)
        self.update_stzcr_section(overlap_length)

        # if 50 ms initialization data are not fullfilled, then keep adding
        if np.sum(np.isnan(self.stat_coarse_section)) > 0:
            self.update_stat_coarse_section(overlap_length)
        elif not self.threshold_found:
            self.threshold_init()  # calculating all threshold values

        self.pointer = self.pointer + overlap_length  # update pointer to the next beginning for STE use

        if np.sum(np.isnan(self.ste_section)) > 0:
            return False
        else:
            return True

    @staticmethod
    def cal_zcr(d):
        """ Calculation of zero crossing rate. """
        if not isinstance(d, np.ndarray):
            d = np.array(d)
        diff = np.sign(d[1:]) - np.sign(d[:-1])
        print('what:{}, d:{}'.format(np.sum(np.abs(diff)), (len(d) * 2)))
        return np.sum(np.abs(diff)) / (len(d) * 2)

    def obtain_zcr(self, d):
        d = d - np.mean(d)
        stzcr_nan_removal = d[(~np.isnan(d)).tolist()]  # remove NaN
        zcr = []
        for i in range(int(len(stzcr_nan_removal)/self.frame_length_10ms)):
            tmp = stzcr_nan_removal[i*self.frame_length_10ms:(i+1)*self.frame_length_10ms]
            zcr.append(self.cal_zcr(tmp))
        return zcr

    def find_fine_start(self):
        lock = True
        zcr_list = np.array(self.obtain_zcr(self.stzcr_section))  # remember that zcr is flipped in [latest ...  oldest]
        tmp = np.flip(np.where(zcr_list - self.threshold.T_zc > 0)[0], axis=0)  # find index of zcr above the threshold and flip around
        if len(tmp) < 3:
            self.fine_start = self.coarse_start
        else:
            for i in range(2, len(tmp)):
                if lock:
                    if tmp[i-2] - tmp[i] == 2:
                        self.fine_start = self.coarse_start - (9-tmp[i])*self.frame_length_10ms
                        lock = False
                    # the last index and still no fine_start is found, then assign coarse_start to it
                    elif i == len(tmp) - 1:
                        self.fine_start = self.coarse_start
                        continue
                elif tmp[i-1] - tmp[i] == 1:
                    self.fine_start = self.fine_start - self.frame_length_10ms
                else:
                    self.fine_start = self.coarse_start
                    break

    def find_fine_end(self, consecutive3, consecutive3flag, i_max10, zcr_value):

        if i_max10 < 10:
            if consecutive3flag:
                if zcr_value[0] > self.threshold.T_zc:
                    self.fine_end = self.pointer
            else:
                if zcr_value[0] > self.threshold.T_zc:  # only one frame, so single value is expeced
                    consecutive3 = consecutive3 + 1
                    if consecutive3 == 3:
                        consecutive3flag = True
            i_max10 = i_max10 + 1
        else:
            self.fine_end_found = True
        return consecutive3, consecutive3flag, i_max10

    def find_coarse_start_end(self, ste_value):

        if not self.coarse_start:
            if (not self.coarse_Tl_found) and (ste_value > self.threshold.T_lower):
                self.coarse_start_candidate = self.pointer
                self.coarse_Tl_found = True
            elif self.coarse_Tl_found and (ste_value < self.threshold.T_lower):
                self.coarse_Tl_found = False
                self.coarse_start_candidate = None
            elif self.coarse_Tl_found and (ste_value > self.threshold.T_upper):
                self.coarse_start = self.coarse_start_candidate - 1 * self.frame_length_10ms  # due to initialization
                self.coarse_Tl_found = False
                self.find_fine_start()
        else:
            if (not self.coarse_Tu_found) and (ste_value < self.threshold.T_upper):
                self.coarse_Tu_found = True
            elif self.coarse_Tu_found and (ste_value < self.threshold.T_lower):
                self.coarse_Tu_found = False
                self.coarse_end = self.pointer

    def audio_activity_detection(self):
        # Loop over the signal
        consecutive3 = 0
        consecutive3flag = False
        i_max10 = 0
        while True:
            if not self.update_sections():
                continue

            ste_value = np.sum(np.abs(self.ste_section)) / 220  # STE value of 10ms data
            zcr_value = self.obtain_zcr(self.ste_section)
            self.check_ste.append(ste_value)
            self.check_zcr.append(zcr_value)

            if (not self.coarse_start or not self.coarse_end) and self.threshold_found:
                self.find_coarse_start_end(ste_value)

            if self.coarse_end:
                consecutive3, consecutive3flag, i_max10 = self.find_fine_end(consecutive3, consecutive3flag, i_max10, zcr_value)

            if self.pointer + (self.frame_length_10ms/2) > len(self.signal_channel_2):
                break

if __name__ == '__main__':

    file_name = 'IN_D1_FWC_S5_20170810_111645.snd'
    fs = 22050  # sampling rate

    kwd = aad(file_name=file_name, fs=fs)  # keyword detection
    kwd.audio_activity_detection()  # find utterance, beginning and ending point

    plt.figure()
    plt.subplot(211)
    plt.plot(kwd.signal_channel_2, 'b')
    plt.plot([kwd.coarse_start, kwd.coarse_start], [-0.1, 0.1], 'r')
    plt.plot([kwd.fine_start, kwd.fine_start], [-0.1, 0.1], 'm')
    plt.plot([kwd.coarse_end, kwd.coarse_end], [-0.1, 0.1], 'r')
    plt.plot([kwd.fine_end, kwd.fine_end], [-0.1, 0.1], 'm')
    plt.xlim([0, 40000])
    plt.grid()
    plt.subplot(212)
    plt.plot(np.array(range(len(kwd.check_ste))) * kwd.frame_length_10ms * 0.5, kwd.check_ste, 'bo')
    plt.plot(np.array(range(len(kwd.check_zcr))) * kwd.frame_length_10ms * 0.5, kwd.check_zcr, 'go-')
    plt.plot([kwd.coarse_start, kwd.coarse_start], [-0.1, 0.1], 'r')
    plt.plot([kwd.fine_start, kwd.fine_start], [-0.1, 0.1], 'm')
    plt.plot([kwd.coarse_end, kwd.coarse_end], [-0.1, 0.1], 'r')
    plt.plot([kwd.fine_end, kwd.fine_end], [-0.1, 0.1], 'm')
    plt.xlim([0, 40000])
    plt.grid()
    plt.show()

