# dlc2_edit_labels
 
Simple editor to create/edit bodypart lables for DeepLabCut

Using the basic framework from maximus009/VideoPlayer<BR>
    https://github.com/maximus009/VideoPlayer


### Interface:
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

