import torch.nn as nn


class YOLO_v1(nn.Module):
    def __init__(self,pre=False):
        super(YOLO_v1, self,).__init__()

        self.pre=pre
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2,padding=7//2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, stride=1, padding=3//2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=2),
            nn.LeakyReLU(0.1),

        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(0.1),

        )
        self.fc=nn.Linear(4*4*1024,1000)
        self.fc1=nn.Linear(7*7*1024,4096)
        self.fc2=nn.Linear(4096,7*7*30)



    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)


        if self.pre:
            x=self.fc1(x)
            x=self.fc2(x)
            x=x.view(-1,7,7,30)
        else:
            x=x.view(-1,4*4*1024)
            x = self.fc(x)

        return x
