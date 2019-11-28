#Processamento de Imagens
#Aluno: JCGG
#Professor:Ricardo de Queiroz 


from __future__ import print_function
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
#print(torch.__version__)
# Training settings

var_loss = 0.9
var_correct = 1693
parser = argparse.ArgumentParser(description='PyTorch Traffic Signal')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('data_sign/GTSRB/Final_Training/Images', transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('data_sign/GTSRB/Final_Test', transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256,400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 43)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=3, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv5(x)),kernel_size=1, stride=2))
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))  
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)
        return F.log_softmax(x)

model = Net()
print(model)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def evaluate():
    model.eval()
    #test_loss = 0
    #correct = 0
    for data,target in evaluate_loader:
        imshow(torchvision.utils.make_grid(data))
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        #test_loss += F.nll_loss(output).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        print('\n Evaluate:',pred)
'''
file_loss = open('record.txt','w')
file_loss.write('epoch, loss, correct\n') 
file_loss.close()
for epoch in range(1, args.epochs + 1):
    file_loss = open('record.txt','a')
    
    train(epoch)
    var, correct = test()
    file_loss.write(str(epoch)+','+str(var)+','+str(correct)+'\n') 
    if var < var_loss or var_correct < correct:
        if var < var_loss:
            var_loss = var
        if var_correct < correct:
           var_correct = correct 
        torch.save(model.state_dict(), 'SignModel'+str(epoch)+'.grafh')

    file_loss.close()