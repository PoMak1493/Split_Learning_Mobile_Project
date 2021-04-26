import socket
import pickle
import torch
from output_layer import output_layer
from server_reply import ServerReply

server_addr = "10.0.2.2"
server_port = 1493


def transmit_layeout(tensor_layout_output, labels, layer_no):
    sockt = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    sockt.settimeout(60)
    conn_result = sockt.connect_ex((server_addr, server_port))
    if conn_result == 0:
        print("Server Port is opening.")
    else:
        print("ServerPort is not open.")
        return None


    #msg = "A message from development client."
    tensor_layout_output_numpy = tensor_layout_output
    #print(tensor_layout_output_numpy)
    outputlayer = output_layer(tensor_layout_output_numpy,  labels, layer_no)


    msg = pickle.dumps(outputlayer)

    msg = msg + bytes("end", encoding='utf8')

    #print(len(msg))

    #The following line is for testing.
    #x = torch.ones(100, 4096)
    #msg = pickle.dumps(x)

    sockt.sendall(msg)
    print("Client sent a message to the server.")


    #rec_data = b''

    recv_data = bytearray()
    while True:
        try:
            packet = sockt.recv(1024)

            #if packet[-3:] == b'end' or not packet:
            #print(packet[-3:] )
            #if packet[-3:] == b'end' or not packet:
            if not packet:
                #packet = packet[:-3]
                #recv_data.extend(packet)
                break

            recv_data.extend(packet)
        except socket.timeout:
            print("The timeout is happened when receiving the server' response.")

        except BaseException as e:

            print("Error from receiving the Android clients' message: {message}.\n".format(message=e))

            return None, 0

    print("The server packet len :")
    print(len(recv_data))

    if recv_data[-3:] == b'end':
        recv_data = recv_data[:-3]
    elif recv_data[-3:] == b'.en':
        recv_data = recv_data[:-2]


    if len(recv_data) > 0:
        try:
            recv_data = pickle.loads(recv_data)

        except pickle.UnpicklingError as ue:
            print("Error again.")
            print(ue)
            return None, 0

    #print("Test3")
    print("Received data from the server: {received_data}".format(received_data=recv_data.get_loss_message()))

    sockt.close()
    print("Socket closed.")

    return recv_data


