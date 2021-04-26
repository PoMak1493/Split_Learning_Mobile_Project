import socket
import pickle

server_addr = "127.0.0.1"
server_port = 1493

sockt = socket.socket();
conn_result = sockt.connect_ex((server_addr, server_port))
if conn_result == 0:
   print("Server Port is opening.")
else:
   print("ServerPort is not open.")


msg = "A message from development client."
msg = pickle.dumps(msg)
sockt.sendall(msg)
print("Client sent a message to the server.")

#rec_data = b''
rec_data = []
while str(rec_data)[-2] != '.':
    data = sockt.recv(4096)
    if not data: break
    rec_data.append(data)

ser_response = pickle.loads(b"".join(rec_data))
print("Received data from the client: {received_data}".format(received_data=ser_response))

sockt.close()
print("Socket closed.")

