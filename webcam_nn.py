import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
#defining the network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16,20, 1)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20*3*3, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        print("Input size", x.size())
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3(x))

        print("Conv.3", x.size())
        
        x = F.max_pool2d(x, (2, 2))

        print("Conv.3 after pooling", x.size())

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cap = cv2.VideoCapture(0) #first webcam in the system
net= Net()  #initialize     constructor
net.load_state_dict(torch.load("cifar_net.pth"))   #eğitilen weightleri içine çekiyor networkün içine yüklüyor 
while(True):
                                                                     # Capture frame-by-frame
    ret, frame = cap.read()  #returned in the frame, return part is for getting true or false   resmi okuduk 
    img = frame
    frame = cv2.resize(frame,(32,32))     #adjusting the size of frame for neural network
    frame_tp = np.transpose(frame, (2,0,1))  # changing the order of variables [batch-channel-height-width] 
    frame_torch = torch.from_numpy(frame_tp).float().cpu() #converting frame numpy to torch
    output = net(frame_torch.unsqueeze(0))     #to expand the dimensions of the frame 3 to 4
    print(output)                         

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   #to get out of the infinite loop 
        break








# When everything done, release the capture
cap.release()   #camera will be released, if we try to use a camera that's in use, we'll get an error
                                        #we can think this as a modifying a file while it is opened
cv2.destroyAllWindows()