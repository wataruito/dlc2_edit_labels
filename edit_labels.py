'''
Orignated from maximus009/VideoPlayer
    https://github.com/maximus009/VideoPlayer

9/14/2020 wi Bug fix
    # freeze_end[epoch,i] = current_frame Identified bug 9/14/2020 wi
    freeze_end[epoch,i] = current_frame - 1

Keyboard controls:
    <Video control>
    w: start palying
    s: stop playing
    a: step back a frame
    d: step forward a frame
    q: play faster
    e: play slower
    r: reset to original inferring coords
    <number>: add bodypart
    <space>: go to next nan

    <Tracking>
    0: drug mode

    <Freezing>
    !: target sub1
    @: target sub2
    j: start freezing
    k: end freezing
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

# global variables
cur_x, cur_y = 100, 100  # current mouse pointer position
drag = False
rclick = False
# sub = ''
pixel_limit = 10.0
mode = 'drag_mode'


def edit_labels(h5_path, video, mag_factor):
    '''
    video_cursor
    '''
    global cur_x, cur_y, drag, rclick, mode, pixel_limit

    ###################################
    # Initialize video windows
    cv2.namedWindow('image')
    cv2.moveWindow('image', 250, 300)
    # Set mouse callback
    cv2.setMouseCallback('image', dragging)
    # Open video file
    cap = cv2.VideoCapture(video)
    # Get the total number of frame
    tots = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Add two trackbars
    cv2.createTrackbar('S', 'image', 0, int(tots)-1, flick)
    cv2.setTrackbarPos('S', 'image', 0)

    cv2.createTrackbar('F', 'image', 1, 100, flick)
    frame_rate = 30
    cv2.setTrackbarPos('F', 'image', frame_rate)
    # cv2.setTrackbarPos('F','image',0)

    ##################################
    # Initialize freeze indicator window for each subject
    sub_freeze = ['sub1_freeze', 'sub2_freeze']

    cv2.namedWindow(sub_freeze[0])
    cv2.moveWindow(sub_freeze[0], 250, 50)

    cv2.namedWindow(sub_freeze[1])
    cv2.moveWindow(sub_freeze[1], 600, 50)

    # Create new blank image
    # freeze
    width, height = 200, 50
    red = (255, 0, 0)
    freeze_sign = create_blank(width, height, rgb_color=red)
    cv2.putText(freeze_sign, "Freeze", (40, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)
    # no_freeze
    green = (0, 255, 0)
    no_freeze_sign = create_blank(width, height, rgb_color=green)
    cv2.putText(no_freeze_sign, "No_freeze", (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, 255)

    ##################################
    # Initialize coordinate display window
    cv2.namedWindow('coords')
    cv2.moveWindow('coords', 1000, 50)

    ###################################

    status_list = {ord('s'): 'stop',
                   ord('w'): 'play',
                   ord('a'): 'prev_frame', ord('d'): 'next_frame',
                   ord('q'): 'slow', ord('e'): 'fast',
                   ord(' '): 'jump_nan',
                   ord('0'): 'drag_mode',
                   ord('!'): 'target_sub1', ord('@'): 'target_sub2',
                   ord('j'): 'start_freezing', ord('k'): 'end_freezing',
                   ord('p'): 'p_value',
                   ord('r'): 'reset_to_original',
                   -1: 'no_key_press',
                   27: 'exit'}
    current_frame = 0
    status = 'stop'
    start_time = time.time()
    real_frame_rate = frame_rate

    # Adjust video width 750 pixel
    _ret, img = cap.read()
    video_format = img.shape
    print("video resolution: {}".format(video_format))
    x_pixcels = img.shape[1]
    y_pixcels = img.shape[0]
    # r = 3
    dim = (x_pixcels*mag_factor, y_pixcels*mag_factor)

    print("total frame number: {}".format(tots))

    length = 5  # cross cursor length

    # prepare to store freezing
    target_freeze = -1
    # freeze_state = False
    freeze_modify = False
    freeze_modify_on_frame = -1
    freeze_modify_off_frame = -1

    # Read DeepLabCut h5 file
    mdf = pd.read_hdf(h5_path)
    mdf_org = mdf.copy()    # Keep original
    # Extract levels
    scorer = mdf.columns.unique(level='scorer').to_numpy()
    individuals = mdf.columns.unique(level='individuals').to_numpy()
    bodyparts = mdf.columns.unique(level='bodyparts').to_numpy()
    coords = mdf.columns.unique(level='coords').to_numpy()
    idx = pd.IndexSlice     # Initialize the IndexSlice
    mdf_modified = np.array([False for x in range(tots)])

    # add member of dictionary
    bodypart_id = 0
    for i_sco in scorer:
        for i_ind in individuals:
            for i_bod in bodyparts:
                bodypart_id += 1
                status_list[ord(str(bodypart_id))] = ['add', i_ind, i_bod]

    # prepare to store trajectory and freezing
    path, filename = os.path.split(video)
    base, _ext = os.path.splitext(filename)
    filename = '_' + base + '_track_freeze.csv'

    if os.path.exists(os.path.join(path, filename)):
        # xy1, xy2, freeze = read_trajectory(video)
        width, half_dep, l1_coord, l2_coord, l4_coord, xy1, xy2, freeze = read_traj(
            video)

        idx = pd.IndexSlice
        bodypart = 'snout'
        coords = ['x', 'y']
        xy1 = mdf.loc[idx[:], idx[:, 'sub1', bodypart, coords]
                      ].to_numpy().astype(int)*mag_factor
        xy2 = mdf.loc[idx[:], idx[:, 'sub2', bodypart, coords]
                      ].to_numpy().astype(int)*mag_factor

    else:
        width = 295.0
        half_dep = 86.5
        l1_coord = [-5, 667]
        l2_coord = [42, 486]
        l4_coord = [914, 670]
        xy1 = np.array([[-1 for x in range(2)] for y in range(tots)])
        xy2 = np.array([[-1 for x in range(2)] for y in range(tots)])
        freeze = np.array([[False for x in range(2)] for y in range(tots)])

    hold_a_bodypart = False
    held_bodypart = ['', '', '']
    p_value = 1.0

    ######################################################################
    # Main loop
    while True:
        try:
            # If reach to the end, play from the begining
            # if current_frame==tots-1:
            if current_frame == tots:
                current_frame = 0

            # read one video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            _ret, img = cap.read()

            # resize video
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # put current state and real frame rate in the image
            im_text1 = "video_status: " + status + ", frame_rate: " + \
                str(real_frame_rate) + " fps"
            im_text2 = "nmode: " + mode + \
                ", target_freeze: " + str(target_freeze+1) + \
                ", freeze_modify: " + str(freeze_modify)

            add_text(img, im_text1, dim[1]-40, 0.5)
            add_text(img, im_text2, dim[1]-20, 0.5)

            ###########################################
            # Display bodyparts marker
            ###########################################
            # Loop all bodyparts
            #   scorer -> individuals -> bodyparts
            for i_sco in scorer:
                for i_ind in individuals:
                    for i_bod in bodyparts:
                        [tab_x, tab_y, likelihood] = mdf.loc[idx[current_frame],
                                                             idx[i_sco, i_ind, i_bod, :]].to_numpy()
                        if not (math.isnan(tab_x) or math.isnan(tab_y)):
                            # Convert the stored bodypart coordinates to match current video size
                            stored_x = int(tab_x)*mag_factor
                            stored_y = int(tab_y)*mag_factor
                            label_deleted = False

            # Drag mouse
                            if not hold_a_bodypart:
                                # print('not hold')
                                # If mouse pointer does not hold any bodypart, check the distance
                                # If less than 10 pixels, then hit
                                if drag and math.sqrt((stored_x-cur_x)**2 +
                                                      (stored_y-cur_y)**2) < pixel_limit:
                                    hold_a_bodypart = True
                                    held_bodypart = [i_sco, i_ind, i_bod]
                                # Store the mouse pointer position into table
                                    mdf.loc[idx[current_frame], idx[i_sco, i_ind, i_bod, :]] = \
                                        [float(cur_x)/mag_factor,
                                         float(cur_y)/mag_factor, likelihood]
                                    mdf_modified[current_frame] = True
                                # Display cross at the mouse pointer position
                                    [dis_x, dis_y] = [cur_x, cur_y]
                                # Display bodypart text on image
                                    cv2.putText(img, i_bod, (dis_x+20, dis_y-20),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                                else:
                                    # Display cross at the store position
                                    [dis_x, dis_y] = [stored_x, stored_y]
                            else:
                                # print('hold')
                                if collections.Counter(held_bodypart) == \
                                        collections.Counter([i_sco, i_ind, i_bod]):
                                    if drag:
                                        # Store the mouse pointer position into table
                                        mdf.loc[idx[current_frame], idx[i_sco, i_ind, i_bod, :]] = \
                                            [float(cur_x)/mag_factor,
                                             float(cur_y)/mag_factor, likelihood]
                                        mdf_modified[current_frame] = True
                                # Display cross at the mouse pointer position
                                        [dis_x, dis_y] = [cur_x, cur_y]
                                # Display bodypart text on image
                                        cv2.putText(img, i_bod, (dis_x+20, dis_y-20),
                                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                                    else:
                                        hold_a_bodypart = False
                                # Display cross at the store position
                                        [dis_x, dis_y] = [stored_x, stored_y]
                                else:
                                    # Display cross at the store position
                                    [dis_x, dis_y] = [stored_x, stored_y]

            # Right click to delete label
                            if rclick and math.sqrt((stored_x-cur_x)**2 +
                                                    (stored_y-cur_y)**2) < pixel_limit:

                                print('one label is deleted')
                                # Store nans into table
                                mdf.loc[idx[current_frame], idx[i_sco, i_ind, i_bod, :]] = \
                                    [math.nan, math.nan, likelihood]
                                label_deleted = True
                                mdf_modified[current_frame] = True

            # Draw cross on video
                            if not label_deleted:
                                if i_ind == 'sub1':
                                    color = (0, 255, 0)
                                elif i_ind == 'sub2':
                                    color = (0, 0, 255)

                                if float(likelihood) < 0.011:
                                    cv2.circle(img, (dis_x, dis_y), 10, color,
                                               thickness=1, lineType=8, shift=0)
                                else:
                                    if float(likelihood) >= p_value:
                                        thickness = 1
                                    else:
                                        thickness = 2

                                    cv2.line(img, (dis_x+length, dis_y+length),
                                             (dis_x-length, dis_y-length), color, thickness)
                                    cv2.line(img, (dis_x+length, dis_y-length),
                                             (dis_x-length, dis_y+length), color, thickness)

            # display freezing state
            if freeze_modify_on_frame == current_frame:
                if target_freeze == -1:
                    freeze_modify_on_frame = -1
                else:
                    freeze_modify = True

            if freeze_modify_off_frame == current_frame:
                if freeze_modify:
                    freeze_modify = False
                    for i in range(freeze_modify_on_frame, freeze_modify_off_frame + 1):
                        freeze[i, target_freeze] = True

                    freeze_modify_on_frame = -1
                    freeze_modify_off_frame = -1

            for i in range(2):
                if freeze_modify and target_freeze == i:
                    # print(i)
                    cv2.imshow(sub_freeze[i], freeze_sign)
                elif freeze[current_frame, i]:
                    cv2.imshow(sub_freeze[i], freeze_sign)
                else:
                    cv2.imshow(sub_freeze[i], no_freeze_sign)

            # display coordinates
            # Create new blank image
            width, height = 400, 180
            # white = (255, 255, 255)
            black = (0, 0, 0)
            coords_blank = create_blank(width, height, rgb_color=black)

            lines_pos = 0
            lines_add = 20
            id_n = 0

            for i_sco in scorer:
                for i_ind in individuals:
                    for i_bod in bodyparts:
                        [tab_x, tab_y, likelihood] = mdf.loc[idx[current_frame],
                                                             idx[i_sco, i_ind, i_bod, :]].to_numpy()
                        if i_ind == 'sub1':
                            color = green
                        if i_ind == 'sub2':
                            color = red

                        lines_pos = lines_pos + lines_add
                        id_n = id_n + 1
                        text = str(id_n)+": "+i_ind+","+i_bod+": "
                        cv2.putText(coords_blank, text, (20, lines_pos),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, tuple(reversed(color)))

                        text = str(tab_x)+"   "+str(tab_y) + \
                            "   "+str(likelihood)
                        cv2.putText(coords_blank, text, (200, lines_pos),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, tuple(reversed(color)))

            cv2.imshow('coords', coords_blank)

            # show video frame
            cv2.imshow('image', img)

            # Read key input
            status_new = status_list[cv2.waitKey(1)]

            if status_new != 'no_key_press':
                status_pre = status
                status = status_new

            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F', 'image')
                if frame_rate == 0.0:
                    continue
                if (time.time() - start_time) > 1.0/frame_rate:
                    real_frame_rate = round(1.0/(time.time() - start_time), 2)
                    current_frame += 1
                    cv2.setTrackbarPos('S', 'image', current_frame)
                    start_time = time.time()
                    continue
            elif status == 'stop':
                current_frame = cv2.getTrackbarPos('S', 'image')
            elif status == 'prev_frame':
                current_frame -= 1
                if current_frame < 0:
                    current_frame = tots-1
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'next_frame':
                current_frame += 1
                if current_frame == tots:
                    current_frame = 0
                cv2.setTrackbarPos('S', 'image', current_frame)
                status = 'stop'
            elif status == 'slow':
                frame_rate = max(frame_rate - 1, 0)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status = status_pre
            elif status == 'fast':
                frame_rate = min(100, frame_rate+1)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status = status_pre
            elif status == 'drag_mode':
                mode = 'drag_mode'
                status = status_pre
            elif status == 'target_sub1':
                target_freeze = 0
                status = status_pre
            elif status == 'target_sub2':
                target_freeze = 1
                status = status_pre
            elif status == 'start_freezing':
                # freeze_state = True
                freeze_modify_on_frame = current_frame
                status = status_pre
            elif status == 'end_freezing':
                # freeze_state = False
                freeze_modify_off_frame = current_frame
                status = status_pre
            elif status == 'jump_nan':
                status = 'stop'
                current_frame = jump_nan(mdf, current_frame)
                cv2.setTrackbarPos('S', 'image', current_frame)
            elif status == 'p_value':
                status = 'stop'
                print("Please input p_value threshould: ", end='')
                p_value = float(input())
            # elif status == 'snap':
            #     cv2.imwrite("./"+"Snap_"+str(i)+".png", img)
            #     print("Snap of Frame", current_frame, "Taken!")
            #     status = 'stop'
            elif status[0] == 'add':
                # print(status[0])
                add_label(mdf, mdf_modified, current_frame,
                          scorer[0], status[1], status[2])
                status = 'stop'
            elif status == 'reset_to_original':
                reset_to_original(mdf, mdf_org, mdf_modified, current_frame)
                status = 'stop'
            elif status == 'exit':
                break

        except KeyError:
            print("Invalid Key was pressed")

    # write file for trajectory and freezing
    write_traj(width, half_dep, l1_coord, l2_coord,
               l4_coord, tots, xy1, xy2, freeze, video)

    # write file for freeze start, end duration
    write_freeze(tots, freeze, video)

    # Write h5 file for extracted frames
    tz_ny = pytz.timezone('America/New_York')
    now = datetime.now(tz_ny)
    extrxt_dir = os.path.join(
        './', now.strftime("%Y%m%d-%H%M%S") + '-extracted')
    if not os.path.isdir(extrxt_dir):
        os.mkdir(extrxt_dir)

    mdf[mdf_modified].to_hdf(extrxt_dir+'/extracted.h5',
                             key='df_output', mode='w')

    # Extract frames modified
    for frame in range(tots):
        if mdf_modified[frame]:
            print("frame = ", frame, " is modified", end=": ")
            # read one video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            _ret, img = cap.read()
            cv2.imwrite(extrxt_dir+"/img" +
                        "{:03d}".format(frame)+".png", img)
            print("Snap of Frame", frame, "Taken!")

    # Clean up windows
    cap.release()
    cv2.destroyAllWindows()


def write_freeze(tots, freeze, video):
    '''
    write_freeze
    '''
    freeze_start = np.array([[-1 for x in range(2)]
                             for y in range(50)], dtype=int)
    freeze_end = np.array([[-1 for x in range(2)]
                           for y in range(50)], dtype=int)
    freeze_dur = np.array([[-1.0 for x in range(2)]
                           for y in range(50)], dtype=float)
    epoch_n = [0, 0]

    for i in range(2):
        freeze_on = False
        epoch = -1
        for current_frame in range(tots):
            if freeze[current_frame, i] and not freeze_on:    # freeze epoch starts
                epoch += 1
                freeze_start[epoch, i] = current_frame
                freeze_on = True
            elif not freeze[current_frame, i] and freeze_on:  # freeze ephoc ends
                # freeze_end[epoch,i] = current_frame Identified bug 9/14/2020 wi
                freeze_end[epoch, i] = current_frame - 1
                freeze_dur[epoch, i] = (
                    freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
                freeze_on = False
        epoch_n[i] = epoch
        if freeze_on:
            freeze_end[epoch, i] = current_frame
            freeze_dur[epoch, i] = (
                freeze_end[epoch, i] - freeze_start[epoch, i] + 1) / 4.0
            freeze_on = False

    column_name = ['start', 'end', 'duration', 'start', 'end', 'duration']
    column_type = ['int', 'int', 'float', 'int', 'int', 'float']

    df_output = pd.DataFrame(
        data=np.concatenate((freeze_start[:, 0][:, np.newaxis],
                             freeze_end[:, 0][:, np.newaxis],
                             freeze_dur[:, 0][:, np.newaxis],
                             freeze_start[:, 1][:, np.newaxis],
                             freeze_end[:, 1][:, np.newaxis],
                             freeze_dur[:, 1][:, np.newaxis]), axis=1),
        columns=column_name)

    # a = dict(zip(column_name, column_type))
    df_output = df_output.astype(dtype=dict(zip(column_name, column_type)))

    # Output to summary.csv
    path, filename = os.path.split(video)
    base, _ext = os.path.splitext(filename)
    filename = '_' + base + '_freeze.csv'

    print("Writing {}".format(filename))
    write_pd2csv(path, filename, df_output, column_name, column_type, 1000)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def reset_to_original(mdf, mdf_org, mdf_modified, current_frame):
    '''
    reset_to_original
    '''
    scorer = mdf.columns.unique(level='scorer').to_numpy()
    individuals = mdf.columns.unique(level='individuals').to_numpy()
    bodyparts = mdf.columns.unique(level='bodyparts').to_numpy()
    idx = pd.IndexSlice     # Initialize the IndexSlice

    for i_sco in scorer:
        for i_ind in individuals:
            for i_bod in bodyparts:
                mdf.loc[idx[current_frame], idx[i_sco, i_ind, i_bod, :]] = \
                    mdf_org.loc[idx[current_frame],
                                idx[i_sco, i_ind, i_bod, :]]

    mdf_modified[current_frame] = False


def jump_nan(mdf, current_frame):
    '''
    jump_nan
    '''
    idx = pd.IndexSlice     # Initialize the IndexSlice
    find = False
    column_nan = np.array(
        [mdf.loc[idx[y], idx[:, :, :, :]].isnull().sum() for y in range(len(mdf.index))])

    if current_frame == len(mdf.index)-1:
        frame = current_frame
    else:
        for frame in range(current_frame+1, len(mdf.index)):
            if column_nan[frame] > 0:
                print('frame '+str(frame)+' contains nan')
                find = True
                break
        if not find:
            for frame in range(0, current_frame+1):
                if column_nan[frame] > 0:
                    print('frame '+str(frame)+' contains nan')
                    find = True
                    break

    return frame


def add_label(mdf, mdf_modified, current_frame, i_sco, i_ind, i_bod):
    '''
    add_label
    '''
    idx = pd.IndexSlice     # Initialize the IndexSlice

    # Store the mouse pointer position into table
    mdf.loc[idx[current_frame], idx[i_sco, i_ind, i_bod, :]] = \
        [100.0, 100.0, 1.0]
    mdf_modified[current_frame] = True
    print('one label is added')


def write_pd2csv(path, filename, df_output, column_name, column_type, _mlw=1000):
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
            output_str = preprocess_output_str(
                output_str, df_output.iloc[i, j], column_type[j], 1000)
        output.write(output_str[0:-1] + '\n')
    output.close()


def preprocess_output_str(output_str, data, column_type, mlw=1000):
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


def dragging(event, read_x, read_y, _flags, _param):
    '''
    dragging

    Mouse events handler
    '''
    global cur_x, cur_y, drag, rclick, mode, pixel_limit
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 'drag_mode':
            drag = True
            cur_x, cur_y = read_x, read_y
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag:
            cur_x, cur_y = read_x, read_y
    elif event == cv2.EVENT_RBUTTONDOWN:
        cur_x, cur_y = read_x, read_y
        rclick = True
    elif event == cv2.EVENT_RBUTTONUP:
        rclick = False


def flick(_x):
    '''
    flick
    '''
    # pass


# def process(img):
#     '''
#     process
#     '''
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def add_text(img, text, text_top, image_scale):
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


def read_traj(video):
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
    # import os
    # import numpy as np
    # import pandas as pd

    # column_name = ['frame', 'sub1_x', 'sub1_y', 'sub2_x',
    #                'sub2_y', 'sub1_freeze', 'sub2_freeze']
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


def write_traj(width, half_dep, l1_coord, l2_coord, l4_coord, tots, xy1, xy2, freeze, video):
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

    # import os
    # import csv
    # import numpy as np
    # import pandas as pd

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
