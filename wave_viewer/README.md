# wave_viewer

Simple viewer for neuronal recorded/processed data, capable to display > 1h data for
  - spectrogram from fieldtrip
  - PAC fA and fP map from BrainStorm
  - LFP and band filtered data from BuzCode

It is scalable using multiprocessing.

#### Not implimented yet
  - synchronize with video player
  - show each frame boundary as time-series
  - show/edit behavioral annotation as time-series


## Interface:
    input_files - specify input files
    window_geo - window size and initial position

    key h: color hotter
    key c: color cooler
    key up: larger time range (shrink)
    key down: smaller time range (magnify)
    key left/right: move viewing window

## Install env:
1. make matplotlib interactive in jupyter lab
```
conda create --name wave_viewer
conda activate wave_viewer
conda install -c conda-forge jupyterlab ipympl nodejs

jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension list 

conda install anaconda
```
2. install the latest hdf5storage
```
git clone https://github.com/frejanordsiek/hdf5storage.git
# Go to the top directory of the repository
pip install -e .
```
3. edit _classic_test_patch.mplstyle<BR>
  reference:<BR>
  jupyter notebook shows error message for matplotlib Bad key “text.kerning_factor”<BR>
  https://stackoverflow.com/questions/61171307/jupyter-notebook-shows-error-message-for-matplotlib-bad-key-text-kerning-factor
