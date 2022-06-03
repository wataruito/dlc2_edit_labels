'''
test for pull request
'''
# from src import wave_viewer as wv
import edit_labels as el
import os


if __name__ == '__main__':

    # input data
    if os.path.exists('input.csv'):
        inferred_video, inferred_h5, labeled_h5, labeled_for_train_pickle = el.read_input(
            'input.csv', 0)
    else:
        ############################
        # example data
        # inferred video
        inferred_video = r'input_data\rpicam-01_1806_20210722_212134.mp4'
        # inferred result h5
        inferred_h5 = r'input_data\rpicam-01_1806_20210722_212134DLC_dlcrnetms5_homecage_test01May17shuffle1_200000_el.h5'
        # labeled data for training
        labeled_h5 = r'input_data\CollectedData_DJ.h5'
        # information which frame is used for training or testing
        labeled_for_train_pickle = r'input_data\Documentation_data-homecage_test01_95shuffle1.pickle'

    # video display magnification factor
    mag_factor = 1
    # set window size and position. win_y_len_axis is only for x-axis window.
    window_geo = {'win_x_len': 1000, 'win_y_len': 100, 'win_y_len_axis': 30,
                  'win_x_origin': 0, 'win_y_origin': 0}

    el.start(inferred_video=inferred_video, inferred_h5=inferred_h5,
             labeled_h5=labeled_h5, labeled_for_train_pickle=labeled_for_train_pickle,
             mag_factor=mag_factor, window_geo=window_geo, plot_type='raster')
