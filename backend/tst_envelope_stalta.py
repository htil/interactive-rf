from radar_utils.helpers import envelope_finder, sta_lta
import numpy as np

fname = '/home/emre/PycharmProjects/ASL_game/data/raw_game_Raw_0_im.png'
prf = 3200  # Hz

up_env, cent_env, low_env, max_velocity = envelope_finder(fname, plot=False)

print('max_velocity: ' + str(max_velocity) + ' m/s')  # max unambiguous: 6.233 m/s

euc_dist = abs(low_env - up_env)

duration = 3
nsta_sec = 0.3
ratio = len(euc_dist) / duration
nsta = int(nsta_sec*ratio)
nlta = int(2*nsta)
stepsz = int(0.2*ratio)  # 0.2
timevec = np.linspace(0, duration, len(euc_dist))
init_th = 0.6
stop_th = 0.2

mask = sta_lta(euc_dist, nlta, nsta, init_th, stop_th, stepsz, duration, plot=True)


