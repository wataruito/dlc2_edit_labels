'''
test for pull request
'''
from wave_viewer import wave_viewer as wv
import edit_labels as el


if __name__ == '__main__':

    # input_h5_path = r'm154DLC_resnet50_test01Dec21shuffle1_100000.h5'
    # input_video = r'm154.mp4'

    input_h5_path = r'rpicam-01_1806_20210722_212134DLC_dlcrnetms5_homecage_test01May17shuffle1_200000_el.h5'
    input_video = r'rpicam-01_1806_20210722_212134.mp4'

    input_mag_factor = 1

    # set window size and position. win_y_len_axis is only for x-axis window.
    window_geo = {'win_x_len': 1000, 'win_y_len': 100, 'win_y_len_axis': 30,
                  'win_x_origin': 0, 'win_y_origin': 0}

    # set input file for each window
    #input_files = [[input_h5_path,     'wave']]
    input_files = [[input_h5_path,     'raster']]

    # start each window
    input_process_list = wv.spawn_wins(input_files, window_geo)

    # open master window for control
    # masterWin = wv.WaveViewerMaster(input_process_list, (0, 20, 1000, 80))
    # masterWin.run()

    masterWin = el.EditLabels(
        input_h5_path, input_video, input_mag_factor, process_list=input_process_list)
    masterWin.edit_labels()

    # wait until all processes stop
    for _process_id_key in input_process_list:
        input_process_list[_process_id_key][0].join()
