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


class WaveViewerMaster():
    '''
    WaveViewerMaster
        Simple example for a master window to command subprocess windows.
    '''

    def __init__(self, process_list, win_geom):
        '''
        __init__
        '''
        self.process_list = process_list
        self.win_geom = win_geom

    def run(self):
        '''
        run
        '''
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

        if event == 'e':
            plt.close(self.fig)

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
        self.t_cur = 10.0       # center, video frame

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
        # Read DeepLabCut h5 file
        self.mdf = pd.read_hdf(self.h5_path)
        # self.mdf_org = self.mdf.copy()    # Keep original
        # Extract data from specific levels
        #   You can access each label by self.individuals[]
        self.scorer = self.mdf.columns.unique(level='scorer').to_numpy()
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
        if self.d_type == 'wave':
            for bodypart in range(len(self.bodyparts)):
                bp_likelihood = self.mdf.loc[self.idx[:],
                                             self.idx[self.scorer[0], self.individuals[0], self.bodyparts[bodypart], self.coords[2]]]
                self.ax_subplot.plot(bp_likelihood)

        if self.d_type == 'x_axis':
            self.ax_plot, = self.ax_subplot.plot(
                self.timestamps[xmin:xmax], np.full(xmax-xmin, 0), linewidth=0)

            plt.subplots_adjust(left=0.05, right=1,
                                bottom=0.99, top=1,
                                wspace=0, hspace=0)
            plt.yticks([])

        self.ax_subplot.set_xlim(t_min, t_max)

    def timer_call_back(self):
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

    def update_plot(self):
        '''
        update_plot
        '''
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


if __name__ == '__main__':

    # set window size and position. win_y_len_axis is only for x-axis window.
    window_geo = {'win_x_len': 1000, 'win_y_len': 100, 'win_y_len_axis': 30,
                  'win_x_origin': 0, 'win_y_origin': 0}

    # set input file for each window
    input_files = [
        [r'm154DLC_resnet50_test01Dec21shuffle1_100000.h5',     'wave']
    ]

    # start each window
    input_process_list = spawn_wins(input_files, window_geo)

    # open master window for control
    masterWin = WaveViewerMaster(input_process_list, (0, 20, 1000, 80))
    masterWin.run()

    # wait until all processes stop
    for _process_id_key in input_process_list:
        input_process_list[_process_id_key][0].join()
