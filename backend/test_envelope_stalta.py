from radar_utils.helpers import envelope_finder
import numpy as np

fname = '/home/ci4r/PycharmProjects/ASL_game/data/raw_game_Raw_0_im.png'
prf = 3200  # Hz

up_env, cent_env, low_env, max_velocity = envelope_finder(fname, plot=False)

print(str(max_velocity) + ' m/s')  # max unambiguous: 6.233 m/s



