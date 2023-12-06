import argparse
import os
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
from dataset import FashionMnist
import torchvision
import utils
from network import LeNet


def get_args_parser():
    parser = argparse.ArgumentParser('FashionMnist', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='LeNet', type=str,
        help="""Name of architecture to train. """)

    # Training/Optimization parameters
    parser.add_argument('--batch_size', default=256, type=int,
        help='batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer.""")

    # Misc
    parser.add_argument('--resume', action="store_true", help='Load checkpoints.')
    parser.add_argument('--output_dir', default="output/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=2, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--log_freq', default=20, type=int, help='Print and save log every x steps.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    
    return parser



class Classify():
    def __init__(self,args):
        self.args = args
        cudnn.benchmark = True
        utils.fix_random_seeds(self.args.seed)
        if not os.path.isdir(self.args.output_dir):
            os.mkdir(self.args.output_dir)


        # ============ preparing data ... ============
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = FashionMnist('train', transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, 
            drop_last=True,
        )
        print(f"Train Data loaded: there are {len(train_dataset)} train images.")

        val_transform = transforms.Compose([transforms.ToTensor()])
        val_dataset = FashionMnist('val', transform=val_transform)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
        )
        print(f"Val Data loaded: there are {len(val_dataset)} val images.")

        # ============ building networks ... ============
        if self.args.arch == 'LeNet':
            net = LeNet()
        else:
            print(f"Unknow architecture: {self.args.arch}")
        self.net = net.cuda()

        # ============ preparing loss ... ============
        self.CELoss = nn.CrossEntropyLoss()


        # ============ preparing optimizer and scheduler ... ============
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr) 
        else:
            print(f"Unknow optimizer: {self.args.optimizer}")
        print(f"Loss, optimizer and schedulers ready.")

        # ============ optionally resume training ... ============ 恢复训练
        to_restore = {"epoch": 0,"acc":0}
        self.best_acc = 0

        if self.args.resume:
            utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            model = self.net
            )
            print("Load from checkpoint ")
        self.start_epoch = to_restore["epoch"]
        self.best_acc = to_restore["acc"]


    def train(self):

        print("Start Training !!!")
        for epoch in range(self.start_epoch,self.args.epochs):
            self.train_one_epoch(epoch)
            self.val(epoch)
        print("Done!!!!,Best ACC is {}%".format(self.best_acc))



    def train_one_epoch(self,epoch):
        self.net.train()
        loss = 0
        train_iter = iter(self.train_loader)
        for batch_idx, (image, label) in enumerate(train_iter):
            image = image.cuda()
            label = label.cuda()
            self.optimizer.zero_grad()

            outputs = self.net(image)
            loss = self.CELoss(outputs, label)
            loss.backward()
            self.optimizer.step()

            if (batch_idx+1) % self.args.log_freq == 0:
                train_log = '[Training] Epoch [{}/{}] Step [{}/{}]  - CE_loss: {:.2f}'. \
                        format(epoch + 1, self.args.epochs, batch_idx+1, len(train_iter), loss.item())
                print(train_log)
                with open(os.path.join(self.args.output_dir,"train_log.txt"),'a') as f_train:
                    f_train.write(train_log+'\n')

    def val(self,epoch):
        print("Validating ...")
        self.net.eval()
        val_iter = iter(self.val_loader)
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(val_iter):
                image = image.cuda()
                label = label.cuda()
                outputs = self.net(image)
                _, predict = outputs.max(1)
                total += image.size(0)
                correct += predict.eq(label).sum().item()
            acc = correct / total * 100
            val_log = 'Epoch {} | ACC: {}% '.format(epoch+1, acc)
            print(val_log)  
            with open(os.path.join(self.args.output_dir,"val_log.txt"),'a') as f_train:
                f_train.write(val_log+'\n')


            save_dict = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch + 1,
                'acc': acc
                }
            if (epoch+1) % self.args.saveckp_freq == 0:
                torch.save(save_dict,os.path.join(self.args.output_dir,'checkpoint.pth'))
                
            if acc > self.best_acc:
                print("Best,Saving")
                self.best_acc = acc
                torch.save(save_dict,os.path.join(self.args.output_dir,'best.pth'))



def main():
    parser = argparse.ArgumentParser('FashionMnist', parents=[get_args_parser()])
    args = parser.parse_args()
    trainer = Classify(args)
    trainer.train()

if __name__ == '__main__':
    main()