'''
test for pull request
'''
from wave_viewer import wave_viewer as wv
import edit_labels as el
import os
import pandas as pd


def read_input(input_csv, i):

    _df = pd.read_csv(input_csv)
    inferred_path = _df.loc[i, 'inferred_path']
    inferred_video = _df.loc[i, 'inferred_video']
    inferred_h5 = _df.loc[i, 'inferred_h5']

    inferred_video = os.path.join(inferred_path, inferred_video)
    inferred_h5 = os.path.join(inferred_path, inferred_h5)

    training_path = _df.loc[i, 'training_path']
    labeled_h5 = _df.loc[i, 'labeled_h5']
    labeled_for_train_pickle = _df.loc[i, 'labeled_for_train_pickle']

    if pd.isna(_df.loc[i, 'training_path']):
        training_path = ''
        labeled_h5 = ''
        labeled_for_train_pickle = ''
    else:
        labeled_h5 = os.path.join(training_path, labeled_h5)
        labeled_for_train_pickle = os.path.join(
            training_path, labeled_for_train_pickle)

    return inferred_video, inferred_h5, labeled_h5, labeled_for_train_pickle


if __name__ == '__main__':

    # input data
    if os.path.exists('input.csv'):
        inferred_video, inferred_h5, labeled_h5, labeled_for_train_pickle = read_input(
            'input.csv', 0)
    else:
        ############################
        # example data
        # inferred video
        inferred_video = r'edit_labels_input_data\rpicam-01_1806_20210722_212134.mp4'
        # inferred result h5
        inferred_h5 = r'edit_labels_input_data\rpicam-01_1806_20210722_212134DLC_dlcrnetms5_homecage_test01May17shuffle1_200000_el.h5'
        # labeled data for training
        labeled_h5 = r'edit_labels_input_data\CollectedData_DJ.h5'
        # information which frame is used for training or testing
        labeled_for_train_pickle = r'edit_labels_input_data\Documentation_data-homecage_test01_95shuffle1.pickle'

    # video display magnification factor
    mag_factor = 1

    # set window size and position. win_y_len_axis is only for x-axis window.
    window_geo = {'win_x_len': 1000, 'win_y_len': 100, 'win_y_len_axis': 30,
                  'win_x_origin': 0, 'win_y_origin': 0}

    # set input file for each window
    input_files = [[inferred_h5,     'wave']]
    input_files = [[inferred_h5,     'raster']]

    # start each window
    input_process_list = wv.spawn_wins(input_files, window_geo)

    masterWin = el.EditLabels(inferred_video=inferred_video, inferred_h5=inferred_h5, mag_factor=mag_factor,
                              labeled_h5=labeled_h5, labeled_for_train_pickle=labeled_for_train_pickle, process_list=input_process_list)
    masterWin.edit_labels()

    # wait until all processes stop
    for _process_id_key in input_process_list:
        input_process_list[_process_id_key][0].join()
