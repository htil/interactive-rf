import time
from radar_v2 import RadarManager
from radar_utils.ioserver import IOServer
from radar_utils.helpers import envelope_finder
import random


sudo_password = 'radar123'
storage_directory = '/home/ci4r/PycharmProjects/ASL_game/data/'
cwd = '/home/ci4r/Desktop/77ghz/CLI/Release'
radar_path = '/home/ci4r/Desktop/77ghz/open_radar/open_radar_initiative-new_receive_test/' \
             'open_radar_initiative-new_receive_test/setup_radar/build'
model_file = '/home/ci4r/PycharmProjects/ASL_game/interactive-rf-main/backend/radar_utils/models/keras_model.h5'

duration = 3  # sec
filename = 'raw_game'
model_im_size = (224, 224)

socketServer = IOServer()
radar_mng = RadarManager(radar_path, cwd, sudo_password, storage_directory)
radar_mng.radar_init()
is_server_init = False


def pred_to_action(pred):
    print('pred_to_action', pred)
    confidence = round(pred[0] * 100 + 0, 2)
    if confidence >= 50:  # if maybe
        # socketServer.send('data', 'Left')
        print("maybe")
    else:  # if you
        fname = storage_directory + filename + '_Raw_0_im.png'
        up_env, cent_env, low_env, max_velocity = envelope_finder(fname, plot=False)
        socketServer.send('speed', max_velocity)

        # socketServer.send('data', 'Right')


def generate_continuous_pred():
    # print('outer')
    eps = 0
    while True:
        try:
            print('loop')
            socketServer.send('recording', "true")

            radar_mng.record_and_plot(filename, duration=duration)
            socketServer.send('recording', "false")

            pred = radar_mng.predict_sample(model_path=model_file, size=model_im_size)
            pred_to_action(pred)
            time.sleep(duration+3)
        except Exception as e:
            # Program throws error sometime if sleep duration is not long enough
            socketServer.send('recording', "false")
            print('some error: ', e)
            radar_mng.hard_stop()


def fire_with_radar():
    print('Speed requested.')
    radar_mng.record_and_plot(filename, duration=duration)
    fname = storage_directory + filename + '_Raw_0_im.png'
    up_env, cent_env, low_env, max_velocity = envelope_finder(fname, plot=False)
    socketServer.send('speed', max_velocity)


def startServer(is_server_init=None):
    if not is_server_init:
        is_server_init = True
        generate_continuous_pred()


socketServer.on("initialize", lambda x: startServer(is_server_init))
# socketServer.on("request_speed", lambda x: fire_with_radar())
socketServer.run()








