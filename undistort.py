import cv2
import numpy as np
import os
import glob
import sys

DIM=(1280, 960)
K=np.array([[652.5161718925217, 0.0, 620.4527637472484], [0.0, 652.7084383029018, 479.1788043987222], [0.0, 0.0, 1.0]])
D=np.array([[-0.19734123540341308], [0.04099901009034639], [-0.03753979105529971], [0.019084545046145075]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, DIM, np.eye(3), balance=0.0), DIM, cv2.CV_16SC2)
def undistort(img):
    return(cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
def video(vid_path):
    cap = cv2.VideoCapture(vid_path)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(vid_path+'_undistorted.mp4', 0x7634706d, 5, (1280, 960))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, length):
        ret, frame = cap.read()
        out.write(undistort(frame))
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return
if __name__ == '__main__':
    for p in sys.argv[1:]:
        video(p)