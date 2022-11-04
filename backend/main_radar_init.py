from radar_v2 import RadarManager


sudo_password = '190396'
storage_directory = '/home/emre/PycharmProjects/ASL_game/data/'
cwd = '/home/emre/Desktop/77ghz/CLI/Release'
radar_path = '/home/emre/Desktop/77ghz/open_radar/open_radar_initiative-new_receive_test/' \
             'open_radar_initiative-new_receive_test/setup_radar/build'

duration = 5  # sec
filename = 'raw_game'

radar_mng = RadarManager(radar_path, cwd, sudo_password, storage_directory)

radar_mng.radar_init()
radar_mng.record_and_plot(filename, duration=duration)








