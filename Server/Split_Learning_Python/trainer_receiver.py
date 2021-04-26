import pickle
import threading
import time
import socket
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch
import split_server_train
import numpy as np
import os.path
from os import path
from server_reply import ServerReply
from split_server_train import Hook
from split_server_train import AlexNet_Server


class SocketThread(threading.Thread):

    def __init__(self, conn, client_addr, buff_size, recv_timeout):
        threading.Thread.__init__(self)
        self.connection = conn
        self.client_addr = client_addr
        self.buff_size = buff_size
        self.recv_timeout = recv_timeout

        # Define Receive Function for the multiple thread message.

    def receive(self):
        recv_data = bytearray()
        # Using while loop to receive the multiple thread message
        while True:
            try:
                packet = self.connection.recv(self.buff_size)

                if packet[-3:] == b'end' or not packet:
                    packet = packet[:-3]
                    recv_data.extend(packet)
                    print("All Client's Data has been received ({received_length}).".format(
                        received_length=len(recv_data)))
                    break

                recv_data.extend(packet)

            except socket.timeout:

                return None, 0

            except BaseException as e:

                print("Error from receiving the Android clients' message: {message}.\n".format(message=e))

                return None, 0

        if len(recv_data) > 0:
            try:
                recv_data = pickle.loads(recv_data)

            except pickle.UnpicklingError as ue:
                return None, 0

            return recv_data, 1

    def run(self):
        while True:
            self.recv_start_time = time.time()
            time_structure = time.gmtime()
            datetime = "Ready to receive data from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(
                year=time_structure.tm_year,
                month=time_structure.tm_mon,
                day=time_structure.tm_mday,
                hour=time_structure.tm_hour,
                minute=time_structure.tm_min,
                second=time_structure.tm_sec)
            print(datetime)
            received_data, status = self.receive()
            if status == 0:
                self.connection.close()
                print(
                    "Connection Closed with: {client_info} caused by the inactivity for {recived_timeout} seconds or due to an error.".format(
                        client_info=self.client_addr, recived_timeout=self.recv_timeout), end="\n\n")
                break

            # (No Multiple Thread Implementation)
            if status == 1:
                #print("The Received Message is : {received_data} ".format(received_data=received_data))


                #The following line is for tesating.
                #test = received_data
                #print(received_data.shape)

                #print(received_data.get_output_layer().shape)
                layer_grad, loss_message = train_in_server(received_data.get_output_layer(), received_data.get_labels(), received_data.get_layer_no())

                layer_grad = layer_grad.cpu().detach()

                #server_reply = ServerReply(layer_grad, loss_message)

                server_reply = ServerReply(layer_grad, loss_message)

                #print(layer_grad)

                reply = pickle.dumps(server_reply,  protocol=5)

                #print(len(reply))


                reply = reply + bytes("end", encoding='utf8')




                #reply = "The trained model has been downloaded."
                self.connection.sendall(reply)
                print("Server sent the message to the Android Client.\n\n")
                self.connection.close()
                break


def train_in_server(outputs_mobile, labels, layer_no):

    outputs_mobile, labels = outputs_mobile.to(device), labels.to(device)

    hook_backward = [Hook(list(net_server._modules.items())[8][1], backward=True)]

    outputs_server = net_server(outputs_mobile, layer_no)

    loss = criterion(outputs_server, labels)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    for hook in hook_backward:
        if hasattr(hook, "input") & hasattr(hook, "output"):
            layer_grad = hook.output[0]

    loss_message = '%.4f' % (loss.item())
    return layer_grad, loss_message



# Server Side
if (path.exists("server_new.pkl") == False):
    net_server = AlexNet_Server()
else:
    net_server = torch.load("server_new.pkl")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_server.parameters(), lr=1e-2, momentum=0.9)
device = torch.device("cuda:0")
net_server.to(device)


# Splitting
server_address = "127.0.0.1"
server_port = 1493
timeout = 30

sockt = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
sockt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print("Socket is established.")

sockt.bind((server_address, server_port))
print("Socket is bound to an address:[{}], Port:[{}].".format(server_address, server_port))

# Listen the android connections
sockt.listen(1)
print("Listening for android connection...")


while True:
    try:
        conn, addr = sockt.accept()
        print("\nConnected to a android client: {client_info}.".format(client_info=addr))
        socket_thread = SocketThread(conn=conn, client_addr=addr, buff_size=4096, recv_timeout=timeout)
        socket_thread.start()
    except socket.timeout:
        sockt.close()
        print("A socket.timeout exception occurred : No connection for {accept_timeout} seconds.".format(accept_timeout=timeout))
        break















