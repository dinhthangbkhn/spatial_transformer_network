import torch 
import model
from torchvision import datasets, transforms
import torch.nn.functional as F 


class Trainer():
    def __init__(self, batch_size=64, download=True, epochs=10):
        self.net = model.Net()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        self.batch_size = batch_size
        self.download = download
        self.device = 'cpu'
        self.epochs = epochs
        self._create_loader()
        self._create_op()
        
    
    def _create_loader(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='.', 
                           train=True, 
                           download=True,         
                           transform=self.transform),
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='.', 
                           train=False, 
                           download=True,         
                           transform=self.transform),
            batch_size=self.batch_size, 
            shuffle=True
        )
        
    def _create_op(self):
        params = self.net.parameters()
        self.train_ops = torch.optim.Adam(params, lr=0.001)
        
    
    def train_one_epoch(self, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.train_ops.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.train_ops.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                
                
    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            
                
    def test(self):
        with torch.no_grad():
            self.net.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)

                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(self.test_loader.dataset),
                          100. * correct / len(self.test_loader.dataset)))

            
            
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()