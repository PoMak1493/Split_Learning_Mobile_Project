import socket, pickle
import numpy as np
from PIL import ImageGrab



while(True):
    HOST = 'localhost'
    PORT = 1493
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    print ('Connected by', addr)

    arr = np.array([[0, 1], [2, 3]])
    printscreen_pil=ImageGrab.grab(bbox=(10,10,500,500))
    img = np.array(printscreen_pil) ## Transform to Array

    data_string = pickle.dumps(img)
    conn.send(data_string)

    msg_recu = conn.recv(4096)
    print(msg_recu)

    conn.close()