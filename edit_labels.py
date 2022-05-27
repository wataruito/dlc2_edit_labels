'''
dlc2_edit_labels

debugging branch :)

Simple editor to create/edit bodypart lables for DeepLabCut

Using the basic framework from maximus009/VideoPlayer
    https://github.com/maximus009/VideoPlayer

Interface:
<Video control>
    w: start palying
    s: stop playing
    a: step back a frame
    d: step forward a frame
    q: play faster
    e: play slower
    <space>: go to next frame containing nan value

<marker manipulation>
    left hold drag: drag a marker
    right click: delete a marker
    r: back to the inferring coords
    <number>: add bodypart (see number for each bod part in the coordinate window)
    p: set p_value, which set the boundary between thick and thin cross marking

<anotating freeze>
    !: target sub1
    @: target sub2
    j: freezing start, freeze_flag on (first video frame for freeze)
    k: freezing end, freeze_flag off (first video frame when animal start moving)
    u: erase freezing annotation, freeze_flag off

<mode change>
    0: drug mode
'''

import os
import time
import math
import collections
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import pytz


class EditLabels():
    '''
    EditLabels
    '''

    def __init__(self, h5_path, video, mag_factor):
        '''
        '''

        self.h5_path = h5_path              # DeepLabCut inferring result file
        self.video = video                  # video path
        self.mag_factor = mag_factor        # magnifying video

        self.status_list = []               # key command list

        self.length = 5                     # cross cursor length
        self.pixel_limit = 10.0
        self.frame_rate = 30
        self.real_frame_rate = self.frame_rate

        self.cur_x, self.cur_y = 100, 100   # current mouse pointer position
        self.current_frame = 0              # current video frame position
        self.status = 'stop'                # current video player status

        self.start_time = time.time()

        self.drag = False
        self.rclick = False
        self.mode = 'drag_mode'             # currently it does not change during operation
        self.hold_a_bodypart = False        # flag holding a bodypart marker with mouse

        # current bodypart id (id_held_bodypart = [i_sco, i_ind, i_bod])
        self.id_held_bodypart = ['', '', '']
        self.p_value = 1.0

        self.freeze_sub = 0                 # target subject # for annotating freeze
        self.freeze_flag = False
        self.freeze_flag_on_frame = -1
        self.freeze_flag_off_frame = -1
        self.freeze_flag_erase_frame = -1
        self.freeze_sub_change = False

        self.black = (0, 0, 0)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)

        self.idx = pd.IndexSlice

        # values will be set in the following code
        self.cap = []
        self.tots = []
        self.dim = []
        self.mdf = []
        self.mdf_org = []
        self.scorer = []
        self.individuals = []
        self.bodyparts = []
        self.coords = []
        self.mdf_modified = []              # specify modified frames for bodypart coordinates

        self.width = []
        self.half_dep = []
        self.l1_coord = []
        self.l2_coord = []
        self.l4_coord = []
        self.xy1 = []
        self.xy2 = []
        self.freeze = []

        self.sub_freeze = []
        self.no_freeze_sign = []
        self.freeze_sign = []

        self.img = []

        self.column_nan = []

    def edit_labels(self, ):
        '''
        video_cursor
        '''
        # global cur_x, cur_y, drag, rclick, mode, pixel_limit

        self.initialize_param()
        self.initialize_windows()

        self.main_loop()

        self.output_files()

        # Clean up windows
        self.cap.release()
        cv2.destroyAllWindows()

    def output_files(self):
        '''
        output_files
        '''
        # write file ([video]_track_freeze.csv) for trajectory and freezing
        self.write_traj(self.width, self.half_dep, self.l1_coord, self.l2_coord,
                        self.l4_coord, self.tots, self.xy1, self.xy2, self.freeze, self.video)

        # write file ([video]_freeze.csv) for freeze start, end duration
        self.write_freeze(self.tots, self.freeze, self.video)

        # outpur h5 file for extracted frames
        tz_ny = pytz.timezone('America/New_York')
        now = datetime.now(tz_ny)
        extrxt_dir = os.path.join(
            './', now.strftime("%Y%m%d-%H%M%S") + '-extracted')
        if not os.path.isdir(extrxt_dir):
            os.mkdir(extrxt_dir)

        self.mdf[self.mdf_modified].to_hdf(extrxt_dir+'/extracted.h5',
                                           key='df_output', mode='w')

        # Extract video frames modified
        for frame in range(self.tots):
            if self.mdf_modified[frame]:
                print("frame = ", frame, " is modified", end=": ")
                # read one video frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                _ret, img = self.cap.read()
                cv2.imwrite(extrxt_dir+"/img" +
                            "{:03d}".format(frame)+".png", img)
                print("Snap of Frame", frame, "Taken!")

    def flick(self, _x):
        '''
        flick
        '''
        # pass

    def mouse_call_back(self, event, read_x, read_y, _flags, _param):
        '''
        dragging

        Mouse events handler
        '''
        # global cur_x, cur_y, drag, rclick, mode, pixel_limit
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'drag_mode':
                self.drag = True
                self.cur_x, self.cur_y = read_x, read_y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = False
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cur_x, self.cur_y = read_x, read_y
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cur_x, self.cur_y = read_x, read_y
            self.rclick = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.rclick = False

    def initialize_windows(self):
        '''
        initialize_windows
        '''
        ###################################
        # Initialize main video windows
        cv2.namedWindow('image')
        cv2.moveWindow('image', 250, 300)
        # Set mouse callback
        cv2.setMouseCallback('image', self.mouse_call_back)

        # Open video file
        self.cap = cv2.VideoCapture(self.video)
        # Get the total number of frame
        self.tots = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Add two slider bars
        # for frame position
        cv2.createTrackbar('S', 'image', 0, int(self.tots)-1, self.flick)
        cv2.setTrackbarPos('S', 'image', 0)
        # for play speed (fps)
        cv2.createTrackbar('F', 'image', 1, 100, self.flick)
        cv2.setTrackbarPos('F', 'image', self.frame_rate)
        # cv2.setTrackbarPos('F','image',0)

        ##################################
        # Initialize freeze indicator window for each subject
        self.sub_freeze = ['sub1_freeze', 'sub2_freeze']
        # animal 1
        cv2.namedWindow(self.sub_freeze[0])
        cv2.moveWindow(self.sub_freeze[0], 250, 50)
        # animal 2
        cv2.namedWindow(self.sub_freeze[1])
        cv2.moveWindow(self.sub_freeze[1], 600, 50)

        # Create image showing freeze/no_freeze
        # freeze
        width, height = 200, 50
        self.freeze_sign = self.create_blank(width, height, rgb_color=self.red)
        cv2.putText(self.freeze_sign, "Freeze", (40, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)
        # no_freeze
        self.no_freeze_sign = self.create_blank(
            width, height, rgb_color=self.green)
        cv2.putText(self.no_freeze_sign, "No_freeze", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)

        ##################################
        # Initialize display window for bodypart coordinate
        cv2.namedWindow('coords')
        cv2.moveWindow('coords', 1000, 50)

        # # generate array for total number of nan value for each video frame
        # self.column_nan = np.array(
        #     [self.mdf.loc[self.idx[y], self.idx[:, :, :, :]].isnull().sum()
        #      for y in range(len(self.mdf.index))])

        # I think this is a better way to do it but we shall see.
        # Above comment contains the old version of this code
        leng = len(self.mdf.loc[1])
        for i in range(0, len(self.mdf)):
            temp = 0
            for j in range(0, leng, 3):
                if math.isnan(self.mdf.loc[i][j]):
                    temp = temp + 1
            self.column_nan.append(temp)

    def initialize_param(self):
        '''
        initialize_param
        '''
        ###################################
        # Open video file
        self.cap = cv2.VideoCapture(self.video)
        # Get the total number of frame
        self.tots = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Adjust video size according to mag_factor
        _ret, img = self.cap.read()
        video_format = img.shape
        x_pixcels = img.shape[1]
        y_pixcels = img.shape[0]
        self.dim = (x_pixcels*self.mag_factor, y_pixcels*self.mag_factor)

        print("video resolution: {}".format(video_format))
        print("total frame number: {}".format(self.tots))

        ###################################
        # Read DeepLabCut h5 file
        self.mdf = pd.read_hdf(self.h5_path)
        self.mdf_org = self.mdf.copy()    # Keep original

        # Extract data from specific levels
        self.scorer = self.mdf.columns.unique(level='scorer').to_numpy()
        self.individuals = self.mdf.columns.unique(
            level='individuals').to_numpy()
        self.bodyparts = self.mdf.columns.unique(level='bodyparts').to_numpy()
        self.coords = self.mdf.columns.unique(level='coords').to_numpy()
        self.mdf_modified = np.array([False for x in range(self.tots)])

        ###################################
        # keyboard commands
        self.status_list = {ord('s'): 'stop',
                            ord('w'): 'play',
                            ord('a'): 'prev_frame', ord('d'): 'next_frame',
                            ord('q'): 'slow', ord('e'): 'fast',
                            ord(' '): 'jump_nan',
                            # ord('0'): 'drag_mode',
                            ord('!'): 'target_sub1', ord('@'): 'target_sub2',
                            ord('j'): 'start_freezing', ord('k'): 'end_freezing',
                            ord('u'): 'erase_freezing',
                            ord('p'): 'p_value',
                            ord('r'): 'reset_to_original',
                            -1: 'no_key_press',
                            27: 'exit'}

        # add member of dictionary for keyboard commands
        bodypart_id = 0
        for _i_sco in self.scorer:
            for i_ind in self.individuals:
                for i_bod in self.bodyparts:
                    bodypart_id += 1
                    self.status_list[ord(str(bodypart_id))] = [
                        'add', i_ind, i_bod]

        ###################################
        # prepare variables to store trajectory and freezing
        # if the file ([video]_track_freeze.csv) already exist, read it
        path, filename = os.path.split(self.video)
        base, _ext = os.path.splitext(filename)
        filename = '_' + base + '_track_freeze.csv'

        if os.path.exists(os.path.join(path, filename)):
            # xy1, xy2, freeze = read_trajectory(video)
            self.width, self.half_dep, self.l1_coord, self.l2_coord, \
                self.l4_coord, self.xy1, self.xy2, self.freeze = self.read_traj(
                    self.video)

            # idx = pd.IndexSlice
            bodypart = 'snout'
            coords = ['x', 'y']
            self.xy1 = self.mdf.loc[self.idx[:], self.idx[:, 'sub1', bodypart, coords]
                                    ].to_numpy().astype(int)*self.mag_factor
            self.xy2 = self.mdf.loc[self.idx[:], self.idx[:, 'sub2', bodypart, coords]
                                    ].to_numpy().astype(int)*self.mag_factor
        # if not exist, create it
        else:
            self.width = 295.0
            self.half_dep = 86.5
            self.l1_coord = [-5, 667]
            self.l2_coord = [42, 486]
            self.l4_coord = [914, 670]
            self.xy1 = np.array([[-1 for x in range(2)]
                                 for y in range(self.tots)])
            self.xy2 = np.array([[-1 for x in range(2)]
                                 for y in range(self.tots)])
            self.freeze = np.array([[False for x in range(2)]
                                    for y in range(self.tots)])

    def disp_marker(self, i_sco, i_ind, i_bod):
        '''
        disp_marker
        '''

        [tab_x, tab_y, likelihood] = self.mdf.loc[self.idx[self.current_frame],
                                                  self.idx[i_sco, i_ind, i_bod, :]].to_numpy()
        # if value is not empty, display the bodypart marker
        if not (math.isnan(tab_x) or math.isnan(tab_y)):
            stored_x = int(tab_x)*self.mag_factor
            stored_y = int(tab_y)*self.mag_factor
            label_deleted = False

        # Dragging a bodypart marker
        # when currently not holding any bodypart
            if not self.hold_a_bodypart:
                # If mouse pointer does not hold any bodypart, check the distance
                # from current mouse pointer coordinate
                # If less than 10 pixels, then hit the current bodypart
                if self.drag and math.sqrt((stored_x-self.cur_x)**2 +
                                           (stored_y-self.cur_y)**2) < self.pixel_limit:
                    self.hold_a_bodypart = True
                    self.id_held_bodypart = [i_sco, i_ind, i_bod]

                    # Store the mouse pointer position into table
                    self.mdf.loc[self.idx[self.current_frame], self.idx[i_sco, i_ind, i_bod, :]] = \
                        [float(self.cur_x)/self.mag_factor,
                            float(self.cur_y)/self.mag_factor, likelihood]
                    self.mdf_modified[self.current_frame] = True

                    # Display cross at the mouse pointer position
                    [dis_x, dis_y] = [self.cur_x, self.cur_y]

                    # Display bodypart text on image
                    #print('coordinate', dis_x+20, dis_y-20)
                    cv2.putText(self.img, i_bod, (dis_x+20, dis_y-20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                else:
                    # Display cross at the store position
                    [dis_x, dis_y] = [stored_x, stored_y]

        # contining bodypart holding
            else:
                # Test if current bodypart is the same to the held bodypart
                if collections.Counter(self.id_held_bodypart) == \
                        collections.Counter([i_sco, i_ind, i_bod]):
                    if self.drag:
                        # Store the mouse pointer position into table
                        self.mdf.loc[self.idx[self.current_frame],
                                     self.idx[i_sco, i_ind, i_bod, :]] = \
                            [float(self.cur_x)/self.mag_factor,
                             float(self.cur_y)/self.mag_factor, likelihood]
                        self.mdf_modified[self.current_frame] = True
                        # Display cross at the mouse pointer position
                        [dis_x, dis_y] = [
                            self.cur_x, self.cur_y]
                        # Display bodypart text on image
                        cv2.putText(self.img, i_bod, (dis_x+20, dis_y-20),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                    else:
                        self.hold_a_bodypart = False
                        # Display cross at the store position
                        [dis_x, dis_y] = [
                            stored_x, stored_y]
                # If different bodypart from the held bodypart
                else:
                    # Display cross at the store position
                    [dis_x, dis_y] = [stored_x, stored_y]

        # Right click to delete the bodypart marker
            if self.rclick and math.sqrt((stored_x-self.cur_x)**2 +
                                         (stored_y-self.cur_y)**2) < self.pixel_limit:

                print('one label is deleted')
                # Store nans into table
                self.mdf.loc[self.idx[self.current_frame], self.idx[i_sco, i_ind, i_bod, :]] = \
                    [math.nan, math.nan, likelihood]
                label_deleted = True
                self.mdf_modified[self.current_frame] = True
                self.column_nan[self.current_frame] = self.column_nan[self.current_frame] + 1

        # Draw circle or cross point as marker on video
            if not label_deleted:
                # differential color for each animal
                if i_ind == 'sub1':
                    color = (0, 255, 0)
                elif i_ind == 'sub2':
                    color = (0, 0, 255)
                # circle for low p value inferred markers
                if float(likelihood) < 0.011:
                    cv2.circle(self.img, (dis_x, dis_y), 10, color,
                               thickness=1, lineType=8, shift=0)
                # thick cross point for >= p_value, thin one for < p_value
                else:
                    if float(likelihood) >= self.p_value:
                        thickness = 1
                    else:
                        thickness = 2

                    cv2.line(self.img, (dis_x+self.length, dis_y+self.length),
                             (dis_x-self.length, dis_y-self.length), color, thickness)
                    cv2.line(self.img, (dis_x+self.length, dis_y-self.length),
                             (dis_x-self.length, dis_y+self.length), color, thickness)

    def freezing_panel(self):
        '''
        # display freezing state panel
        '''
        # at the video frame for entering freezing, set self.freeze_flag True
        if self.freeze_flag_on_frame == self.current_frame:
            if self.freeze_sub == -1:
                self.freeze_flag_on_frame = -1
            else:
                self.freeze_flag = True

        # at the video frame for exiting freezing (frame where animal starts to move),
        # set the flag False, and set True for the freezing duration
        if self.freeze_flag_off_frame == self.current_frame:
            if self.freeze_sub == -1:
                self.freeze_flag_off_frame = -1

            elif self.freeze_flag:
                self.freeze_flag = False
                for i in range(self.freeze_flag_on_frame, self.freeze_flag_off_frame):
                    self.freeze[i, self.freeze_sub] = True

                self.freeze_flag_on_frame = -1
                self.freeze_flag_off_frame = -1

        # whenever erase is pressed, freeze_flag reset (previous freeze annotation is lost),
        # remove True value from the self.freeze
        if self.freeze_flag_erase_frame == self.current_frame:
            if self.freeze_sub == -1:
                self.freeze_flag_off_frame = -1
            else:
                self.freeze_flag = False
                self.freeze_flag_on_frame = -1
                self.freeze_flag_off_frame = -1
                self.freeze[self.current_frame, self.freeze_sub] = False

        # whenever subject changed, freeze_flag reset (previous freeze annotation is lost)
        if self.freeze_sub_change:
            self.freeze_flag = False
            self.freeze_flag_on_frame = -1
            self.freeze_flag_off_frame = -1
            self.freeze_sub_change = False

        # display premade panel image on freezing panel
        for i in range(2):
            if self.freeze_flag and self.freeze_sub == i:
                #print('freeze', i)
                cv2.imshow(self.sub_freeze[i], self.freeze_sign)
            elif self.freeze[self.current_frame, i]:
                #print('freeze', i)
                cv2.imshow(self.sub_freeze[i], self.freeze_sign)
            else:
                #print('no_freeze', i)
                cv2.imshow(self.sub_freeze[i], self.no_freeze_sign)

    def coordinate_panel(self):
        '''
        coordinate_panel
        '''
        # display coordinates on coordinate panel
        # Create new blank image
        width, height = 400, 180
        # white = (255, 255, 255)
        black = (0, 0, 0)
        coords_blank = self.create_blank(width, height, rgb_color=black)

        lines_pos = 0
        lines_add = 20
        id_n = 0

        for i_sco in self.scorer:
            for i_ind in self.individuals:
                for i_bod in self.bodyparts:
                    [tab_x, tab_y, likelihood] = self.mdf.loc[self.idx[self.current_frame],
                                                              self.idx[i_sco, i_ind, i_bod, :]].to_numpy()
                    if i_ind == 'sub1':
                        color = self.green
                    if i_ind == 'sub2':
                        color = self.red

                    lines_pos = lines_pos + lines_add
                    id_n = id_n + 1
                    text = str(id_n)+": "+i_ind+","+i_bod+": "
                    cv2.putText(coords_blank, text, (20, lines_pos),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, tuple(reversed(color)))

                    text = str(tab_x)+"   "+str(tab_y) + \
                        "   "+str(likelihood)
                    cv2.putText(coords_blank, text, (200, lines_pos),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, tuple(reversed(color)))
        # display on the panel
        cv2.imshow('coords', coords_blank)

    def key_comm(self, status_new):
        '''
        key_comm
        '''

        if status_new != 'no_key_press':
            status_pre = self.status
            self.status = status_new

        # video play mode
        if self.status == 'play':
            self.frame_rate = cv2.getTrackbarPos('F', 'image')

            # when frame_rate is 0, do nothing
            if self.frame_rate == 0.0:
                pass
            # count up frame number only when it passese more than 1.0/frame_rate
            elif (time.time() - self.start_time) > 1.0/self.frame_rate:
                self.real_frame_rate = round(
                    1.0/(time.time() - self.start_time), 2)
                self.current_frame += 1
                cv2.setTrackbarPos('S', 'image', self.current_frame)
                self.start_time = time.time()

        # video stop mode
        elif self.status == 'stop':
            self.current_frame = cv2.getTrackbarPos('S', 'image')

        # goto previous frame
        elif self.status == 'prev_frame':
            self.current_frame -= 1
            if self.current_frame < 0:
                self.current_frame = self.tots-1
            cv2.setTrackbarPos('S', 'image', self.current_frame)
            self.status = 'stop'

        # goto next frame
        elif self.status == 'next_frame':
            self.current_frame += 1
            if self.current_frame == self.tots:
                self.current_frame = 0
            cv2.setTrackbarPos('S', 'image', self.current_frame)
            self.status = 'stop'

        # slow down playing speed
        elif self.status == 'slow':
            self.frame_rate = max(self.frame_rate - 1, 0)
            cv2.setTrackbarPos('F', 'image', self.frame_rate)
            self.status = status_pre

        # speed up playing speed
        elif self.status == 'fast':
            self.frame_rate = min(100, self.frame_rate+1)
            cv2.setTrackbarPos('F', 'image', self.frame_rate)
            self.status = status_pre

        # elif self.status == 'drag_mode':
        #     self.mode = 'drag_mode'
        #     status = status_pre

        # target for anotating freeze to sub1
        elif self.status == 'target_sub1':
            self.freeze_sub = 0
            self.freeze_sub_change = True
            self.status = status_pre

        # target for anotating freeze to sub2
        elif self.status == 'target_sub2':
            self.freeze_sub = 1
            self.freeze_sub_change = True
            self.status = status_pre

        # annotate start_freezing
        elif self.status == 'start_freezing':
            # freeze_state = True
            self.freeze_flag_on_frame = self.current_frame
            self.status = status_pre

        # annotate end_freezing
        elif self.status == 'end_freezing':
            # freeze_state = False
            self.freeze_flag_off_frame = self.current_frame
            self.status = status_pre

        # erase freeze annotate
        elif self.status == 'erase_freezing':
            # freeze_state = False
            self.freeze_flag_erase_frame = self.current_frame
            self.status = status_pre

        # go to the next frame with nan value
        elif self.status == 'jump_nan':
            self.status = 'stop'
            self.current_frame = self.jump_nan()
            cv2.setTrackbarPos('S', 'image', self.current_frame)

        # set the p_value to change thickness of cross marker
        elif self.status == 'p_value':
            self.status = 'stop'
            print("Please input p_value threshould: ", end='')
            self.p_value = float(input())

        # snap the current frame to file
        # elif status == 'snap':
        #     cv2.imwrite("./"+"Snap_"+str(i)+".png", img)
        #     print("Snap of Frame", current_frame, "Taken!")
        #     status = 'stop'

        # add bodypart marker which is missing
        elif self.status[0] == 'add':
            # print(status[0])
            self.add_label(self.scorer[0], self.status[1], self.status[2])
            self.status = 'stop'

        # go back to original coordiante for a bodypart marker which is moved
        elif self.status == 'reset_to_original':
            self.reset_to_original()
            self.status = 'stop'

        elif self.status == 'exit':
            return True

        return False

    def jump_nan(self):
        '''
        jump_nan
        '''
        # idx = pd.IndexSlice     # Initialize the IndexSlice

        find = False

        # current video frame position is at the end of video, do nothing
        if self.current_frame == len(self.mdf.index)-1:
            frame = self.current_frame
        # current position in the middle of vidoe
        else:
            # scan from current position to the end of video
            for frame in range(self.current_frame+1, len(self.mdf.index)):
                if self.column_nan[frame] > 0:
                    print('frame '+str(frame)+' contains nan')
                    find = True
                    break
            # scan from the beginning to the current position
            if not find:
                for frame in range(0, self.current_frame+1):
                    if self.column_nan[frame] > 0:
                        print('frame '+str(frame)+' contains nan')
                        # find = True
                        break
        return frame

    def reset_to_original(self):
        '''
        reset_to_original
        '''
        for i_sco in self.scorer:
            for i_ind in self.individuals:
                for i_bod in self.bodyparts:
                    self.mdf.loc[self.idx[self.current_frame], self.idx[i_sco, i_ind, i_bod, :]] = \
                        self.mdf_org.loc[self.idx[self.current_frame],
                                         self.idx[i_sco, i_ind, i_bod, :]]

        self.mdf_modified[self.current_frame] = False

    def add_label(self, i_sco, i_ind, i_bod):
        '''
        add_label
        '''
        # Store the mouse pointer position into table
        self.mdf.loc[self.idx[self.current_frame], self.idx[i_sco, i_ind, i_bod, :]] = \
            [self.cur_x, self.cur_y, 1.0]
        self.mdf_modified[self.current_frame] = True
        print('one label is added')
        self.column_nan[self.current_frame] = self.column_nan[self.current_frame] - 1

    def main_loop(self):
        '''
        Main loop
        '''
        while True:
            try:
                # If reach to the end, play from the begining
                # if current_frame==tots-1:
                if self.current_frame == self.tots:
                    self.current_frame = 0

                # read a video frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                _ret, self.img = self.cap.read()

                # resize video
                self.img = cv2.resize(self.img, self.dim,
                                      interpolation=cv2.INTER_AREA)

                # display current state and real frame rate in the video
                im_text1 = "video_status: " + self.status + ", frame_rate: " + \
                    str(self.real_frame_rate) + " fps"
                im_text2 = "nmode: " + self.mode + \
                    ", freeze_sub: sub-" + str(self.freeze_sub+1) + \
                    ", freeze_flag: " + str(self.freeze_flag)

                # add_text(self.img, im_text1, self.dim[1]-40, 0.5)
                # add_text(self.img, im_text2, self.dim[1]-20, 0.5)
                self.add_text(self.img, im_text1, self.dim[1]-40, 0.5)
                self.add_text(self.img, im_text2, self.dim[1]-20, 0.5)

                # Display markers for each bodyparts
                # Loop for all bodyparts
                #   scorer -> individuals -> bodyparts
                for i_sco in self.scorer:
                    for i_ind in self.individuals:
                        for i_bod in self.bodyparts:
                            self.disp_marker(i_sco, i_ind, i_bod)

                # show video frame
                cv2.imshow('image', self.img)

                # display freezing state panel
                self.freezing_panel()

                # display coordinates and p_value on coordinate panel
                self.coordinate_panel()

                # keyborad command
                # Read key input
                status_new = self.status_list[cv2.waitKey(1)]
                if self.key_comm(status_new):
                    break

            except KeyError:
                print("Invalid Key was pressed")

    def create_blank(self, width, height, rgb_color=(0, 0, 0)):
        """
        Create new image(numpy array) filled with certain color in RGB
        """
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    def add_text(self, img, text, text_top, image_scale):
        """
        Args:
            img (numpy array of shape (width, height, 3): input image
            text (str): text to add to image
            text_top (int): position of top text to add
            image_scale (float): image resize scale

        Summary:
            Add display text to a frame.

        Returns:
            Next available position of top text (allows for chaining this function)
        """
        cv2.putText(
            img=img,
            text=text,
            org=(0, text_top),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=image_scale,
            color=(0, 255, 255),
            thickness=2)
        return text_top + int(5 * image_scale)

    def read_traj(self, video):
        '''
        # read *_trac_freeze.csv file and extract
        #    landmark coordinates for l1_coord, l2_coord, and l4_coord
        #    trajectory coordinates for sub1 and sub2
        #    freezing state (bool) for sub1 and sub2
        #
        # <file format>
        # measurement:
        # l1_coord-l4_coord(width), 295.0
        # l1_coord-l2_coord(half_dep), 86.5
        #
        # landmark:
        # name,x,y
        # l1_coord, ,
        # l2_coord, ,
        # l4_coord, ,
        #
        # coordinate:
        # frame,sub1_x,sub1_y,sub2_x,sub2_y,sub1_freeze,sub2_freeze
        #
        # Old format, which starts with frame,sub1_x ... can be read.
        #
        '''
        column_type = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']

        # defalt values
        width = 295.0
        half_dep = 86.5
        l1_coord, l2_coord, l4_coord = [0, 0], [0, 0], [0, 0]

        path, filename = os.path.split(video)
        base, _ext = os.path.splitext(filename)
        filename = '_' + base + '_track_freeze.csv'
        input_filename = os.path.join(path, filename)

        print("Reading {}".format(filename))

        # reading csv file
        with open(input_filename, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)

            # extracting each data row one by one
            for row in csvreader:
                if row[0] == 'L1-L4(width)':
                    width = float(row[1])
                elif row[0] == 'L1-L2(half_dep)':
                    half_dep = float(row[1])
                elif row[0] == 'L1':
                    l1_coord = [int(row[1]), int(row[2])]
                elif row[0] == 'L2':
                    l2_coord = [int(row[1]), int(row[2])]
                elif row[0] == 'L4':
                    l4_coord = [int(row[1]), int(row[2])]
                elif row[0] == 'coordinate:':
                    break
                elif row[0] == 'frame':
                    csvfile.seek(csvreader.line_num - 1)  # back one line
                    break
            # after break, use dataframe.read_csv
            df_input = pd.read_csv(csvfile, index_col=False)
            # Need to convert to object to set numpy array in a cell
            df_input = df_input.astype(object)

        # Post process from str to array
        for i in range(0, len(df_input)):
            for j in range(0, len(df_input.columns)):
                if column_type[j] == 'int_array':
                    df_input.iloc[i, j] = np.fromstring(
                        df_input.iloc[i, j][1:-1], dtype=int, sep=' ')

        xy1 = df_input[['sub1_x', 'sub1_y']].to_numpy()
        xy2 = df_input[['sub2_x', 'sub2_y']].to_numpy()
        freeze = df_input[['sub1_freeze', 'sub2_freeze']].to_numpy()

        return width, half_dep, l1_coord, l2_coord, l4_coord, xy1, xy2, freeze

    def write_traj(self, width, half_dep, l1_coord, l2_coord, l4_coord, tots, xy1, xy2, freeze, video):
        '''
        # write *_trac_freeze.csv file
        #
        # <file format>
        # measurement:
        # L1-L4(width), 295.0
        # L1-L2(half_dep), 86.5
        #
        # landmark:
        # name,x,y
        # L1, ,
        # L2, ,
        # L4, ,
        #
        # coordinate:
        # frame,sub1_x,sub1_y,sub2_x,sub2_y,sub1_freeze,sub2_freeze
        #
        '''

        # Initialize Pandas DataFrame
        column_name = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
                       'sub2_y', 'sub1_freeze', 'sub2_freeze']
        column_type = ['int', 'int', 'int', 'int', 'int', 'bool', 'bool']

        # Need 2D matrix to tranpose
        frame_num = np.array(list(range(tots)))[np.newaxis]
        frame_num = np.transpose(frame_num)
        df_output = pd.DataFrame(data=np.concatenate(
            (frame_num, xy1, xy2, freeze), axis=1), columns=column_name)
        df_output = df_output.astype(dtype=dict(zip(column_name, column_type)))

        # Output to summary.csv
        path, filename = os.path.split(video)
        base, _ext = os.path.splitext(filename)
        filename = '_' + base + '_track_freeze.csv'
        output_filename = os.path.join(path, filename)

        print("Writing {}".format(filename))

        with open(output_filename, 'w', newline='') as csvfile:  # newline='' is for windows
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(['measurement:'])
            spamwriter.writerow(['L1-L4(width)', width])
            spamwriter.writerow(['L1-L2(half_dep)', half_dep])
            spamwriter.writerow([''])
            spamwriter.writerow(['landmark:'])
            spamwriter.writerow(['name', 'x', 'y'])
            spamwriter.writerow(['L1', l1_coord[0], l1_coord[1]])
            spamwriter.writerow(['L2', l2_coord[0], l2_coord[1]])
            spamwriter.writerow(['L4', l4_coord[0], l4_coord[1]])
            spamwriter.writerow([''])
            spamwriter.writerow(['coordinate:'])

            df_output.to_csv(csvfile, index=False)

    def write_freeze(self, tots, freeze, video):
        '''
        write_freeze
        '''
        epoch_num = 50

        freeze_start = np.array([[-1 for x in range(2)]
                                 for y in range(epoch_num)], dtype=int)
        freeze_end = np.array([[-1 for x in range(2)]
                               for y in range(epoch_num)], dtype=int)
        freeze_dur = np.array([[-1.0 for x in range(2)]
                               for y in range(epoch_num)], dtype=float)
        epoch_total_num = [0, 0]

        for i in range(2):  # for two subjects
            freeze_on = False
            epoch = -1      # counting epoch

            # scan all video frames
            for current_frame in range(tots):

                # freeze epoch starts
                if freeze[current_frame, i] and not freeze_on:
                    epoch += 1
                    freeze_start[epoch, i] = current_frame
                    freeze_on = True

                # freeze ephoc ends
                elif not freeze[current_frame, i] and freeze_on:
                    freeze_end[epoch, i] = current_frame - 1
                    freeze_dur[epoch, i] = (
                        freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
                    freeze_on = False

            epoch_total_num[i] = epoch

            # if the last video frame is freeze, make things clean
            if freeze_on:
                freeze_end[epoch, i] = current_frame
                freeze_dur[epoch, i] = (
                    freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
                freeze_on = False

        column_name = ['start', 'end', 'duration', 'start', 'end', 'duration']
        column_type = ['int', 'int', 'float', 'int', 'int', 'float']

        # store into pandas dataframe
        df_output = pd.DataFrame(
            data=np.concatenate((freeze_start[:, 0][:, np.newaxis],
                                 freeze_end[:, 0][:, np.newaxis],
                                 freeze_dur[:, 0][:, np.newaxis],
                                 freeze_start[:, 1][:, np.newaxis],
                                 freeze_end[:, 1][:, np.newaxis],
                                 freeze_dur[:, 1][:, np.newaxis]), axis=1),
            columns=column_name)

        # set each dtype
        df_output = df_output.astype(dtype=dict(zip(column_name, column_type)))

        # Output to summary.csv
        path, filename = os.path.split(video)
        base, _ext = os.path.splitext(filename)
        filename = '_' + base + '_freeze.csv'

        print("Writing {}".format(filename))
        self.write_pd2csv(path, filename, df_output,
                          column_name, column_type, 1000)

    def write_pd2csv(self, path, filename, df_output, column_name, column_type, _mlw=1000):
        '''
        write_pd2csv
        '''
        # import os
        # import numpy as np
        # import pandas as pd

        output_filename = os.path.join(path, filename)
        output = open(output_filename, "w")
        # mlw = 1000 # max_line_width in np.array2string

        output.write(','.join(column_name)+'\n')

        for i in range(0, len(df_output)):
            output_str = ''
            for j in range(0, len(column_name)):
                # print(df_output.shape,j)
                output_str = self.preprocess_output_str(
                    output_str, df_output.iloc[i, j], column_type[j], 1000)
            output.write(output_str[0:-1] + '\n')
        output.close()

    def preprocess_output_str(self, output_str, data, column_type, mlw=1000):
        '''
        preprocess_output_str
        '''
        # import numpy as np

        if column_type == 'int_array':
            output_str = output_str + \
                np.array2string(data, max_line_width=mlw) + ','
        # elif column_type == 'float':
        elif column_type == 'float' or column_type == 'bool':
            # print(data)
            output_str = output_str + str(data) + ','
        elif column_type == 'str':
            output_str = output_str + data + ','
        elif column_type == 'int':
            output_str = output_str + str(data) + ','

        return output_str


if __name__ == '__main__':

    input_h5_path = r'rpicam-01_1806_20210722_212134DLC_dlcrnetms5_homecage_test01May17shuffle1_200000_el.h5'
    input_video = r'rpicam-01_1806_20210722_212134.mp4'
    input_mag_factor = 1

    el = EditLabels(input_h5_path, input_video, input_mag_factor)
    el.edit_labels()
