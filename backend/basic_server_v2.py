from radar_utils.ioserver import IOServer

socketServer = IOServer()


def sendMessageToClient(message):
    socketServer.send('data', message)


def clientConnected():
    print("Client connected")

    # Eventually we want to replace this call with a command from the radar signal processing pipeline
    sendMessageToClient("Left")


socketServer.on("initialize", lambda x: clientConnected())

socketServer.run()

print(socketServer)
