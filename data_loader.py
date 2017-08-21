import os
import soundfile
import matplotlib.pyplot as plt


def parse_data(file_name):
    """ Decode or retrieve data

    Args:
        file_name: file name

    Returns:
        audio data
    """

    current_dir = os.getcwd() + '\\audio_data'
    data_path = os.path.join(current_dir, file_name)
    with open(data_path, 'rb') as mysf:
        sound_data = soundfile.read(mysf,channels=2, samplerate=22050, subtype='PCM_16', endian='BIG', format='RAW')
    return sound_data[0]
if __name__ == '__main__':
    file_name = 'IN_D1_FWC_S5_20170810_111645.snd'
    fs = 22050  # sampling rate
    sound_data = parse_data(file_name)



    plt.figure(1)
    plt.subplot(211)
    plt.plot(sound_data[:,0])
    plt.grid()
    plt.subplot(212)
    plt.plot(sound_data[:,1])
    plt.grid()
    plt.show()