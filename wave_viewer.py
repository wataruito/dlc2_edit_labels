'''
wave-viewer.py

Simple viewer for neuronal recorded/processed data
    Capable to display > 1h data for
        spectrogram from fieldtrip
        PAC fA and fP map from BrainStorm
        LFP and band filtered data from BuzCode

Interface:
    input_files - specify input files
    window_geo - window size and initial position

    h: color hotter
    c: color cooler
    up: larger time range (shrink)
    down: smaller time range (magnify)
    left/right: move viewing window
'''
import math
import sys
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PyQt5 import QtCore
import hdf5storage
import pandas as pd
import os


class WaveViewerMaster():
    '''
    WaveViewerMaster
        Simple example for a master window to command subprocess windows.
    '''

    def __init__(self, process_list, win_geom, h5_path):
        '''
        __init__
        '''
        self.process_list = process_list
        self.win_geom = win_geom

        self.h5_path = h5_path

        self.t_cur = 0.0       # current video frame

    def run(self):
        '''
        run
        '''

        ###################################
        # Read DeepLabCut h5 file
        self.mdf = pd.read_hdf(self.h5_path)
        # self.mdf_org = self.mdf.copy()    # Keep original
        # Extract data from specific levels
        #   You can access each label by self.individuals[]
        # self.scorer = self.mdf.columns.unique(level='scorer').to_numpy()
        # self.individuals = self.mdf.columns.unique(
        #     level='individuals').to_numpy()
        # self.bodyparts = self.mdf.columns.unique(level='bodyparts').to_numpy()
        # self.coords = self.mdf.columns.unique(level='coords').to_numpy()
        # # self.mdf_modified = np.array([False for x in range(self.tots)])
        self.max_time = len(self.mdf)

        # create window
        mpl.rcParams['toolbar'] = 'None'    # need to put this to hide toolbar
        self.fig = plt.figure()
        # set callback for key_press_event as self.press function
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.window().statusBar().setVisible(False)

        # move window position and remove title bar
        self.mngr = plt.get_current_fig_manager()
        self.mngr.window.setGeometry(*self.win_geom)

        plt.show()

    def key_press(self, event):
        '''
        key_press
        '''
        # print('press', event.key)
        sys.stdout.flush()

        if hasattr(event, 'key'):
            event = event.key

        if event == 'd':
            self.t_cur += 1
            if self.t_cur == self.max_time:
                self.t_cur = 0
        if event == 'a':
            self.t_cur -= 1
            if self.t_cur < 0:
                self.t_cur = self.max_time

        if event == 'e':
            plt.close(self.fig)

        if event in ['d', 'a']:
            # send pressed key and the extent to each process
            for _process_id_key in self.process_list:
                self.process_list[_process_id_key][1].put(self.t_cur)
        else:
            # send pressed key and the extent to each process
            for _process_id_key in self.process_list:
                self.process_list[_process_id_key][1].put(event)

        # wait for completion of task
        for _process_id_key in self.process_list:
            self.process_list[_process_id_key][1].join()


class WaveViewer(multiprocessing.Process):
    '''
    WaveViewer
        Subordinate windows
    '''

    def __init__(self, task_queue, result_queue, h5_path, d_type, win_geom):
        '''
        __init__
        '''
        # for multiprocessing
        multiprocessing.Process.__init__(self)

        self.task_queue = task_queue
        self.result_queue = result_queue

        # data to show as grap
        self.h5_path = h5_path
        self.d_type = d_type

        # pandas index
        self.idx = pd.IndexSlice

        # matplotlib
        self.fig = []
        self.ax_subplot = []
        self.ax_plot = []
        self.mngr = []

        # windows setting
        self.x_axis = False     # toggle x-axis
        self.win_geom = win_geom
        self.orig_x, self.orig_y = win_geom[0], win_geom[1]

        # initial viewing window
        self.t_width = 20.0     # width, video frame
        self.t_cur = 0.0       # center, video frame

        # current line
        self.cur_line = {'wave': [0.0, 1.0], 'raster': [-0.5, 7.5]}

        # for contour plot, like spectrogram
        self.hmin, self.hmax = 0.0, 0.0
        self.color_fac = 1.0
        self.wave_data = []
        self.timestamps = []
        self.spec_freq = []

    def run(self):
        '''
        run
        '''
        ###################################
        # Read h5 file for inferred coords
        self.mdf = pd.read_hdf(self.h5_path)
        # self.mdf_org = self.mdf.copy()    # Keep original
        # Extract data from specific levels
        #   You can access each label by self.individuals[]
        self.scorers = self.mdf.columns.unique(level='scorer').to_numpy()
        self.individuals = self.mdf.columns.unique(
            level='individuals').to_numpy()
        self.bodyparts = self.mdf.columns.unique(level='bodyparts').to_numpy()
        self.coords = self.mdf.columns.unique(level='coords').to_numpy()
        # self.mdf_modified = np.array([False for x in range(self.tots)])
        self.max_time = len(self.mdf)

        ###################################
        # Create matplotlib windows
        self.create_window()
        # Plot data
        if self.d_type in ('spec', 'paca', 'pacp'):
            # self.disp_2d()
            pass
        else:
            self.disp_1d()

        ###################################
        # start timer for receiving message from master window
        timer = self.fig.canvas.new_timer(interval=10)
        timer.add_callback(self.timer_call_back)
        timer.start()

        plt.show()

    def create_window(self):
        '''
        create_window
        '''
        # create window
        mpl.rcParams['toolbar'] = 'None'    # need to put here to hide toolbar
        self.fig = plt.figure()
        # key input in the local window
        self.fig.canvas.mpl_connect(
            'key_press_event', self.local_key_call_back)
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.window().statusBar().setVisible(False)

        # move window position and remove title bar
        self.mngr = plt.get_current_fig_manager()
        # self.mngr.window.setGeometry(*self.win_geom)
        # self.mngr.window.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        # create subplot
        self.ax_subplot = plt.subplot()

        # Define plotting area. It determine either show up x-axis and labels or not
        # bottom = 0.0  # DEBUG for x-scale. Nomal is bottom = 0
        bottom = 0.2  # DEBUG for x-scale. Nomal is bottom = 0
        plt.subplots_adjust(left=0.05, right=1,
                            bottom=bottom, top=1,
                            wspace=0, hspace=0)

        # show glids
        plt.grid(ls='--', lw=0.25)

    def disp_1d(self):
        '''
        disp_1d
        '''
        # # compute view window size
        t_min = self.t_cur - self.t_width/2
        t_max = self.t_cur + self.t_width/2

        # plot wave or x_axis
        colors = ['g', 'r']
        if self.d_type == 'wave':
            for individual in range(len(self.individuals)):

                for bodypart in range(len(self.bodyparts)):
                    bp_likelihood = self.mdf.loc[self.idx[:],
                                                 self.idx[self.scorers[0], self.individuals[individual],
                                                          self.bodyparts[bodypart], self.coords[2]]]
                    bp_likelihood = np.nan_to_num(bp_likelihood, nan=0.0)

                    self.ax_subplot.plot(bp_likelihood, colors[individual])

            self.lines = self.ax_subplot.plot(
                [self.t_cur, self.t_cur], self.cur_line[self.d_type], 'k--')

        if self.d_type == 'raster':

            events = self.comp_likelihood_threshold(lh_threshold=0.1)

            colors = np.array(['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r'])

            self.ax_subplot.eventplot(
                np.flip(events, 0), linelengths=0.8, colors=np.flip(colors))

            self.lines = self.ax_subplot.plot(
                [self.t_cur, self.t_cur], [-0.5, 7.5], 'k--')

        if self.d_type == 'x_axis':
            self.ax_plot, = self.ax_subplot.plot(
                self.timestamps[xmin:xmax], np.full(xmax-xmin, 0), linewidth=0)

            plt.subplots_adjust(left=0.05, right=1,
                                bottom=0.99, top=1,
                                wspace=0, hspace=0)
            plt.yticks([])

        self.ax_subplot.set_xlim(t_min, t_max)

    def comp_likelihood_threshold(self, lh_threshold=0.1):
        # compute index arrays the likelihood is below the threshold
        # lh_threshold = 0.1
        _events = {}
        for scorer in self.scorers:
            for individual in self.individuals:
                for bodypart in self.bodyparts:
                    _a = self.mdf[(scorer, individual, bodypart,
                                   self.coords[2])].isnull().to_numpy()
                    _b = (self.mdf[(scorer, individual, bodypart,
                                    self.coords[2])] < lh_threshold).to_numpy()
                    _events[individual +
                            bodypart] = self.mdf[np.logical_or(_a, _b)].index.to_numpy()
        events = list(_events.values())

        # print out the ratio of Nan or below the threshold to the total video frames
        print('## The ratio of Nan to the entire video frames. (total: ',
              self.max_time, ' frames)')
        for individual in self.individuals:
            for bodypart in self.bodyparts:
                print('{0}: {1:8.2f}'.format(individual + bodypart,
                                             len(_events[individual + bodypart]) / self.max_time))

        return events

    def timer_call_back(self):
        '''
        timer_call_back
        '''
        if not self.task_queue.empty():
            # next_task = self.task_queue.get()
            self.t_cur = self.task_queue.get()

            if self.t_cur == 'e':
                self.cmd_interp(self.t_cur)
            else:
                # self.t_cur = self.t_width/16.0

                self.update_plot()

            # elif event == 'up':
            #     self.t_width = self.t_width * 2.0
            # elif event == 'down':
            #     self.t_width = self.t_width / 2.0

            # view window cannot be bigger than max_time
            if self.t_width > self.max_time:
                self.t_width = self.max_time
            # # center cannot move beyond max_time
            # if self.t_cur + self.t_width/2.0 > self.max_time:
            #     self.t_cur = self.max_time - self.t_width/2.0
            # # center cannot move below 0
            # if self.t_cur - self.t_width/2.0 < 0.0:
            #     self.t_cur = 0.0 + self.t_width/2.0

            # if event in ('right', 'left', 'up', 'down'):
            #     self.update_plot()
            # elif event == 'm':
            #     self.move_window()
            # else:
            #     self.cmd_interp(event)
            #     self.update_plot()

            self.task_queue.task_done()

        return True

    def timer_call_back_org(self):
        '''
        timer_call_back
        '''
        if not self.task_queue.empty():
            # next_task = self.task_queue.get()
            event = self.task_queue.get()

            shift = self.t_width/16.0

            if event == 'right':
                self.t_cur = self.t_cur + shift
            elif event == 'left':
                self.t_cur = self.t_cur - shift
            elif event == 'up':
                self.t_width = self.t_width * 2.0
            elif event == 'down':
                self.t_width = self.t_width / 2.0

            # view window cannot be bigger than max_time
            if self.t_width > self.max_time:
                self.t_width = self.max_time
            # center cannot move beyond max_time
            if self.t_cur + self.t_width/2.0 > self.max_time:
                self.t_cur = self.max_time - self.t_width/2.0
            # center cannot move below 0
            if self.t_cur - self.t_width/2.0 < 0.0:
                self.t_cur = 0.0 + self.t_width/2.0

            if event in ('right', 'left', 'up', 'down'):
                self.update_plot()
            elif event == 'm':
                self.move_window()
            else:
                self.cmd_interp(event)
                self.update_plot()

            self.task_queue.task_done()
        return True

    def local_key_call_back(self, event):
        '''
        local_key_call_back
        '''
        sys.stdout.flush()

        if hasattr(event, 'key'):
            event = event.key
        if event != 'e':
            self.cmd_interp(event)
            self.update_plot()

    def find_nearest(self, array, value):
        '''
        find_nearest
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def cmd_interp(self, event):
        '''
        cmd_interp
            local command list
        '''
        if event == 'h':  # hotter
            self.hmax = self.hmax / self.color_fac
        elif event == 'c':  # cooler
            self.hmax = self.hmax * self.color_fac
        elif event == 'x':
            self.x_axis = not self.x_axis
        elif event == 'e':
            plt.close(self.fig)
        elif event == 'up':
            self.t_width = self.t_width * 2.0
        elif event == 'down':
            self.t_width = self.t_width / 2.0

        # view window cannot be bigger than max_time
        if self.t_width > self.max_time:
            self.t_width = self.max_time

    def update_plot(self):
        '''
        update_plot
        '''

        self.lines.pop(0).remove()

        self.lines = self.ax_subplot.plot(
            [self.t_cur, self.t_cur], self.cur_line[self.d_type], 'k--')

        # compute extent
        t_min = self.t_cur - self.t_width/2
        t_max = self.t_cur + self.t_width/2

        if self.d_type in ('spec', 'paca', 'pacp'):
            # extent = [t_min, t_max, 0, 200]
            extent = [t_min, t_max, math.log(
                self.spec_freq[0], 2.0), math.log(self.spec_freq[-1], 2.0)]

            self.ax_plot.set_data(self.wave_data[:, xmin:xmax])
            self.ax_plot.set_clim(self.hmin, self.hmax)
            self.ax_plot.set_extent(extent)
            self.ax_subplot.set_xlim(t_min, t_max)

        else:
            self.ax_subplot.set_xlim(t_min, t_max)
            # self.ax_subplot.autoscale_view(True, True, True)

        if self.d_type != 'x_axis':
            if self.x_axis:
                bottom = 0.2
            else:
                bottom = 0.2
            plt.subplots_adjust(left=0.05, right=1,
                                bottom=bottom, top=1,
                                wspace=0, hspace=0)

        self.fig.canvas.draw()

    def move_window(self):
        '''
        move_window
        '''
        geom = self.mngr.window.geometry()
        _, _, x_len, y_len = geom.getRect()

        self.mngr.window.setGeometry(self.orig_x, self.orig_y, x_len, y_len)

        self.fig.canvas.draw()


def spawn_wins(process_members, window_spec):
    '''
    spawn_wins
    '''
    # decode window size/position setting
    win_x_len = window_spec['win_x_len']
    win_y_len = window_spec['win_y_len']
    win_y_len_axis = window_spec['win_y_len_axis']  # only for x-axis window
    win_x_origin = window_spec['win_x_origin']
    win_y_origin = window_spec['win_y_origin']

    process_list = {}
    process_list_num = 0
    for process in process_members:
        win_y_origin = win_y_origin + win_y_len
        if process[1] == 'x_axis':
            input_tuple = tuple(process) + \
                ((win_x_origin, win_y_origin, win_x_len, win_y_len_axis),)
        else:
            input_tuple = tuple(process) + \
                ((win_x_origin, win_y_origin, win_x_len, win_y_len),)

        # print(input_tuple)

    # for process in process_members:
        task = multiprocessing.JoinableQueue()
        result = multiprocessing.Queue()

        process_id = WaveViewer(task, result, *input_tuple)
        process_id.start()
        print('Started: ', process_id)

        process_list_num += 1
        process_list[str(process_list_num)] = (process_id, task, result)

    return process_list


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
        # inferred result h5
        inferred_h5 = r'edit_labels_input_data\rpicam-01_1806_20210722_212134DLC_dlcrnetms5_homecage_test01May17shuffle1_200000_el.h5'

    # set window size and position. win_y_len_axis is only for x-axis window.
    window_geo = {'win_x_len': 1000, 'win_y_len': 100, 'win_y_len_axis': 30,
                  'win_x_origin': 0, 'win_y_origin': 0}

    # set input file for each window
    input_files = [[inferred_h5,     'wave']]
    input_files = [[inferred_h5,     'raster']]

    # start each window
    input_process_list = spawn_wins(input_files, window_geo)

    # open master window for control
    masterWin = WaveViewerMaster(
        input_process_list, (0, 20, 1000, 80), inferred_h5)
    masterWin.run()

    # wait until all processes stop
    for _process_id_key in input_process_list:
        input_process_list[_process_id_key][0].join()
