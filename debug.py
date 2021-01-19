# import pandas as pd
import edit_labels as vp
h5_path = r'W:\videos_synchrony\20200713\m154DLC_resnet50_test01Dec21shuffle1_100000.h5'
# df = pd.read_hdf(path)


video = r'm154.mp4'

vp.edit_labels(h5_path, video, 2)
