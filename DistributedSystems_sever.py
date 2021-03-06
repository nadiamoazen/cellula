# proj0-s
from socket import *

serverPort = 12001
import threading


#building routing table and storing data
arrays_ip = []
arrays_id = []
arrays = []
arrays_data=[]
import socket


def get_Host_name_IP():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print("Hostname :  ", host_name)
        print("IP : ", host_ip)
        # s='Grad01.acs.uwinnipeg.ca'  # print(''.join(i for i in host_name if i.isdigit()))
        Host_name = ''.join(i for i in host_name if i.isdigit())
        return host_ip, Host_name
    except:
        print("Unable to get Hostname and IP")


identity = get_Host_name_IP()
print("get_Host_name_IP()", identity[0], identity[1])
identity_str = identity[0] + " " + identity[1]

arrays_id.append(identity[1])
arrays_ip.append(identity[0])


class myThread(threading.Thread):
    def __init__(self, threadIP, msg):
        threading.Thread.__init__(self)
        self.threadIP = threadIP
        self.msg = msg

        string = threadIP# getting IP address
        mylist = string.split('.')# split by "."
        print(mylist[3])#last digit would be identifier
        lstD = ''.join(i for i in mylist[3] if i.isdigit())#chanaging a list to int
        self.name = int(lstD) - 20

    def run(self):
        print("\nNode IP:", self.threadIP, "Received msg: ", self.msg, "self.name", self.name)

        text = str(self.msg)
        D_hash = ''.join(i for i in text if i.isdigit())
        arrays_data.append(self.msg)
        arrays_data.append(D_hash)

        if "join" in text:
            if not self.threadIP in arrays_ip:
                arrays_ip.append(self.threadIP)
                text1 = ''.join(i for i in text if i.isdigit())
                arrays_id.append(text1)
                print("arrays_ip", arrays_ip)
                print("arrays_id", arrays_id)
                print("arrays_data", arrays_data)

                print_ring(arrays)
            # for i in range(0, len(arrays_ip)):
            #    print("node", i, "=>", end="")
            else:
                print("\n Alreaady existed")
                print_ring(arrays)

    # print_time(self.name, self.threadID)


def print_ring(arrays):
    # print("Full potential ring")
    for i in range(0, len(arrays_ip)):
        # print(type(i))
        # print(threading.currentThread().getName(), '->', end="")
        print("node", i, "->", end="")
    print("\n")


from socket import *

serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)

serverSocket.bind(('', serverPort))
print('The udp server is ready to receive...')

while True:
    message, clientAddress = serverSocket.recvfrom(2048)
    # print("hello")
    print(message.decode(), clientAddress)
    # print('got message in udp...')
    print("clientAddress:", clientAddress)
    #generating thread for later communication
    thread1 = myThread(clientAddress[0], message.decode())
    arrays.append(thread1)
    thread1.start()
    modifiedMessage = message.decode().upper()
    # print(modifiedMessage)
    print("len(arrays)", len(arrays))
    if len(arrays) == 1:
        print("len(arrays)", len(arrays))
        modifiedMessage = identity[0]
        serverSocket.sendto(modifiedMessage.encode(), clientAddress)
        print("my Successor is node", arrays[0].name, "with Ip:", arrays[0].threadIP)
        print("my Precessor is node", arrays[0].name, "with Ip:", clientAddress[0])
    elif len(arrays) > 1:
        print("my Successor is node", arrays[0].name, "with Ip:", arrays[0].threadIP)

        string = clientAddress[0]
        mylist = string.split('.')
        print(mylist[3])
        lstD = ''.join(i for i in mylist[3] if i.isdigit())
        pre_num = int(lstD) - 20

        print("my Precessor is node", pre_num, "with Ip:", clientAddress[0])
        print("len(arrays)", len(arrays))
        print("arrays_ip[1]", arrays_ip[1])
        modifiedMessage = arrays[0].threadIP

        serverSocket.sendto(modifiedMessage.encode(), clientAddress)
    # serverSocket.sendto(modifiedMessage.encode(), clientAddress)
