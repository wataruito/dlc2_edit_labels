# dlc2_edit_labels
 
Simple editor to create/edit bodypart lables for DeepLabCut
- Read a inferred output (like *_resnet50_test01Dec21shuffle1_100000.h5) and movie (*.mp4)
- Display video and markers for inferred bodyparts<BR>
    circle <0.011, thick cross < (adjustable, defalut=0.9) > thin cross
- The markers can be dragged freely
- Add manual annotation for freezing for each mouse
- Output only the frames and bodypart coordinates that are modified

#### Not implimented yet
- Merge with existing training dataset and relabel them if need

Using the basic code from maximus009/VideoPlayer<BR>
    https://github.com/maximus009/VideoPlayer


## Procedures:
#### Step0: Annotate freezing

    Read video
    |
    Annotate freezing
    |
    Output trajectory and freezing file (*_trajectory_freezing.csv, *_freezing.csv)


#### Step1: Scan bad prediction from the entire video

    Read DeepLabCut inferring output, Video
    |
    Select frame with false prediction
    |
    Output dlc2_edit_labels output file
     frame#, coordinates, description (history etc.)

#### Step2: Merge the selected bad frame with the previous training dataset

    Read DeepLabCut training dataset, like “CollectedData_wataru.h5”
    |
    Merge into the internal dlc2_edit_labels data frame
     frame#, coordinates, description (history etc.)
    |
    Add/drop frame and edit coordinate
    |
    Output
    1. dlc2_edit_labels output file
     frame#, coordinates, description (history etc.)|
    2. DeepLabCut training dataset, like “CollectedData_wataru.h5”
    3. corresponding video frames

## Interface:
#### video control
    w: start palying
    s: stop playing
    a: step back a frame
    d: step forward a frame
    q: play faster
    e: play slower
    <space>: go to next frame containing nan value

#### marker manipulation
    (left hold drag): drag a marker
    (right click): delete a marker
    r: back to the inferring coords
    <number>: add bodypart (see number for each bodypart in the coordinate window)
    p: set p_value, which set the boundary between thick and thin cross marking

#### annotate freezing
    !: target sub1
    @: target sub2
    j: freezing start, freeze_flag on (first video frame for freeze)
    k: freezing end, freeze_flag off (first video frame when animal start moving)
    u: erase freezing annotation, freeze_flag off

#### mode change
    0: drug mode

## Install env:

