import time
from radar_v2 import RadarManager
from radar_utils.ioserver import IOServer
import random


sudo_password = '190396'
storage_directory = '/home/emre/PycharmProjects/ASL_game/data/'
cwd = '/home/emre/Desktop/77ghz/CLI/Release'
radar_path = '/home/emre/Desktop/77ghz/open_radar/open_radar_initiative-new_receive_test/' \
             'open_radar_initiative-new_receive_test/setup_radar/build'
model_file = '/home/emre/PycharmProjects/RadarGUI/keras_model.h5'

duration = 5  # sec
filename = 'raw_game'
model_im_size = (224, 224)

socketServer = IOServer()
radar_mng = RadarManager(radar_path, cwd, sudo_password, storage_directory)
# radar_mng.radar_init()
# radar_mng.record_and_plot(filename, duration=duration)
pred = radar_mng.predict_sample(model_path=model_file, size=model_im_size)


def pred_to_server(eps):
    for i, p in enumerate(pred):
        confidence = round(p * 100 + eps, 2)
        socketServer.send('data', 'Class #' + str(i+1) + ' confidence: ' + str(confidence) + '%')


def generate_continuous_pred():
    while True:
        eps = random.randint(0, 10)
        print(eps)
        pred_to_server(eps)
        time.sleep(1)


socketServer.on("initialize", lambda x: generate_continuous_pred())
socketServer.run()








