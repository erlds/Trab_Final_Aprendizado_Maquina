#Processamento de Imagens
#Aluno: JCGG
#Professor:Ricardo de Queiroz 


from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


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
        self.fc3 = nn.Linear(400, 3)

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
model.load_state_dict(torch.load('SignModel763.grafh'))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def evaluate():
    
    evaluate_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('prova/eval', transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    shuffle= True)    
    
      
    for data,target in evaluate_loader:    
        import datetime
        imshow(torchvision.utils.make_grid(data))
        a = datetime.datetime.now()
        #img = imread(infile)        
        #plt.show()
        data = Variable(data, volatile=True)
        output = model(data)
        #test_loss += F.nll_loss(output).data[0]
        pred = output.data.max(1)[1][0][0] # get the index of the max log-probability
        tex = ''
        #print(pred[0][0])
        if pred == 2:
            tex = 'spray'
        elif pred == 0:
            tex = 'cc'
        elif pred == 1:
            tex = 'globular'
        print('\n Evaluate:',tex)
        b = datetime.datetime.now()
    
        print(b-a)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('prova/test', transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    shuffle=True)
    
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k) 
    
def mapk(actual, predicted, k):
    #return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    return np.mean([apk(actual,predicted,k)])
    
def test():
    #model.eval()
    test_loss = 0
    correct = 0
    top5count = 0
    target_list =[]
    pred_list = []
    for data, target in test_loader:
        
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        #print(target.data[0])
        pred_list.append(pred[0][0])
        target_list.append(target.data[0])
        
        correct += pred.eq(target.data).cpu().sum()
        
        top5Indexs = sorted(range(len(pred_list)), key=lambda x: pred_list[x])[-5:]
        #print(top5Indexs)
        for index in top5Indexs:
            if (int(index) == int(target.data[0])):
                top5count += 1
                break


    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #map    
    val_map = mapk(target_list,pred_list,k=43)
    print('\nTest set: Main Average Pression: {:.2f}, top-5 error: {:.2f}%\n'.format(
        val_map, 100. *top5count / len(test_loader.dataset)))

if __name__ == "__main__":
    
    
    evaluate()
    #test()