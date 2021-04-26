import socket, pickle
import numpy as np

HOST = 'localhost'
PORT = 1523
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

msg_a_envoyer = "hello".encode()
s.sendall(msg_a_envoyer)

data = []
while True:
    packet = s.recv(4096)
    if not packet: break
    data.append(packet)
data_arr = pickle.loads(b"".join(data))
print (data_arr)
s.close()