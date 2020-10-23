import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time, math, json, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from args import get_train_args
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class EarlyStopping:
    """
    Early stopping
    patience: conter가 patience 값 이상으로 쌓이면 early stop
    min_delta: 최상의 score 보다 낮더라도 conter를 증가시키지 않도록 하는 margin 값
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = np.NINF

    def __call__(self, loss):
        score = -1 * loss
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            return False

def main():
    early_stopping = EarlyStopping(5, 0.0)
    opts = get_train_args()
    print ("load data ...")
    train_data = datasets.ImageFolder(root="data/train",
                           transform=transforms.Compose([
                               transforms.Resize((256,256)),       # 한 축을 128로 조절하고
                               #transforms.CenterCrop(256),  # square를 한 후,
                               transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s 니까...
                           ]))

    valid_data = datasets.ImageFolder(root="data/val",
                                      transform=transforms.Compose([
                                          transforms.Resize((256,256)),  # 한 축을 128로 조절하고
                                          #transforms.CenterCrop(128),  # square를 한 후,
                                          transforms.ToTensor(),  # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                                          transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 사이로 normalize
                                                               (0.5, 0.5, 0.5)),  # (c - m)/s 니까...
                                      ]))
    train_loader = DataLoader(train_data,
                             batch_size=opts.batch_size,
                             shuffle=True,
                             num_workers=opts.num_processes)

    valid_loader = DataLoader(valid_data,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=opts.num_processes)

    classes = train_data.classes
    print (classes)

    print ("load model ...")
    if opts.model == 'resnet':
        model = models.resnet50(progress=True)
    elif opts.model == 'vggnet':
        model = models.vgg13_bn(progress=True)
    elif opts.model == 'googlenet':
        model = models.googlenet(progress=True)
    elif opts.model == 'densenet':
        model = models.densenet121(progress=True)
    else:
        model = models.resnext50_32x4d(progress=True)
    print (opts.model)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    model.cuda()
    loss = torch.nn.CrossEntropyLoss()
    batch_nums = np.round(14400/opts.batch_size)
    valid_nums = np.round(1600/opts.batch_size)

    print ("start training")
    for epoch in range(1, opts.epochs + 1):
        print ("epoch : " + str(epoch))
        model.train()
        epoch_loss = 0
        tot = 0
        cnt = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.cuda(), labels.cuda()
            train_loss = loss(model(inputs), labels)
            train_loss.backward()
            optimizer.step()
            batch_loss = train_loss.item()
            epoch_loss += batch_loss
            cnt += 1
            print ('\r{:>10} epoch {} progress {} loss: {}\n'.format('', epoch, tot/14400, train_loss))

        with open(str(opts.model)+' log.txt', 'a') as f:
            f.write(str(epoch) + ' loss : ' + str(epoch_loss/batch_nums) + '\n')
        model.eval()
        valid_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                batch_loss = loss(outputs, labels)
                batch_loss = batch_loss.item()
                valid_loss += batch_loss
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            acc = 100 * correct / total


        with open(str(opts.model)+' log.txt', 'a') as f:
            f.write(str(epoch) + ' loss : ' + str(valid_loss/valid_nums) + ' acc : ' + str(acc) + '\n')


        # check early stopping
        if early_stopping(valid_loss):
            print("[Training is early stopped in %d Epoch.]" % epoch)
            torch.save(model.state_dict(), 'model.pt')
            print("[Saved the trained model successfully.]")
            break

        if epoch % opts.save_step == 0:
            print ("save model...")
            torch.save(model.state_dict(), 'model.pt')

    print("save model...")
    model.entity_normalize()
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main()