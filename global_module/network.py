import torch
from torch import nn
import math

import sys
sys.path.append('./global_module/')
from attontion import PAM_Module, CAM_Module,PAM_X_Module,PAM_Y_Module
from activation import mish, gelu, gelu_new, swish


####################################################################################################
####################################################################################################
#                                        OUR NET                                                   #
####################################################################################################
####################################################################################################

class TBTA_dense2net(nn.Module):
    def __init__(self, band, classes):
        super(TBTA_dense2net, self).__init__()

        self.name = 'TBTA_dense2net'
        self.scales=4

        self.conv_feature=nn.Sequential(nn.Conv3d(in_channels=1, out_channels=24,kernel_size=(1, 1, 7), stride=(1, 1, 2)),
                                        nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),   
                                        mish()) 

        # spectral Branch
        self.conv111 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv112=nn.Sequential(nn.Conv3d(6,6,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv113=nn.Sequential(nn.Conv3d(6,6,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv114=nn.Sequential(nn.Conv3d(6,6,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv115 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        



        self.conv121 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv122=nn.Sequential(nn.Conv3d(12,12,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv123=nn.Sequential(nn.Conv3d(12,12,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv124=nn.Sequential(nn.Conv3d(12,12,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv125 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())



        self.conv131 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv132=nn.Sequential(nn.Conv3d(24,24,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv133=nn.Sequential(nn.Conv3d(24,24,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv134=nn.Sequential(nn.Conv3d(24,24,padding=(0, 0, 3),kernel_size=(1, 1, 7), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv135 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())


        kernel_3d = math.ceil((band - 6) / 2)

        self.conv14 =nn.Sequential(nn.Conv3d(in_channels=192, out_channels=192, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(192,eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish()
                                    )

        # Spatial Branch x
        self.conv211 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv212=nn.Sequential(nn.Conv3d(6,6,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv213=nn.Sequential(nn.Conv3d(6,6,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv214=nn.Sequential(nn.Conv3d(6,6,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv215 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        

        self.conv221 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv222=nn.Sequential(nn.Conv3d(12,12,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv223=nn.Sequential(nn.Conv3d(12,12,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv224=nn.Sequential(nn.Conv3d(12,12,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv225 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())



        self.conv231 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv232=nn.Sequential(nn.Conv3d(24,24,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv233=nn.Sequential(nn.Conv3d(24,24,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv234=nn.Sequential(nn.Conv3d(24,24,padding=(1, 0, 0),kernel_size=(3, 1, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv235 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())



        self.conv24 =nn.Sequential(nn.Conv3d(in_channels=192, out_channels=192, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(192,eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish()
                                    )


        # Spatial Branch y
        self.conv311 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv312=nn.Sequential(nn.Conv3d(6,6,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv313=nn.Sequential(nn.Conv3d(6,6,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv314=nn.Sequential(nn.Conv3d(6,6,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(6,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv315 =nn.Sequential( nn.Conv3d(24,24,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        

        self.conv321 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv322=nn.Sequential(nn.Conv3d(12,12,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv323=nn.Sequential(nn.Conv3d(12,12,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv324=nn.Sequential(nn.Conv3d(12,12,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(12,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv325 =nn.Sequential( nn.Conv3d(48,48,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(48,eps=0.001, momentum=0.1, affine=True),
                                    mish())



        self.conv331 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv332=nn.Sequential(nn.Conv3d(24,24,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv333=nn.Sequential(nn.Conv3d(24,24,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())
        self.conv334=nn.Sequential(nn.Conv3d(24,24,padding=(0, 1, 0),kernel_size=(1, 3, 1), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(24,eps=0.001, momentum=0.1, affine=True),
                                    mish())

        self.conv335 =nn.Sequential( nn.Conv3d(96,96,kernel_size=(1,1,1)),
                                    nn.BatchNorm3d(96,eps=0.001, momentum=0.1, affine=True),
                                    mish())



        self.conv34 =nn.Sequential(nn.Conv3d(in_channels=192, out_channels=192, padding=(0, 0, 0),kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)),
                                    nn.BatchNorm3d(192,eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    mish()
                                    )



        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(192,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial_x = nn.Sequential(
                                    nn.BatchNorm3d(192,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial_y = nn.Sequential(
                                    nn.BatchNorm3d(192,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(576, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(192)
        self.attention_spatial_x = PAM_X_Module(192)
        self.attention_spatial_y = PAM_Y_Module(192)

    def forward(self, X):
        X = self.conv_feature(X) # n*24*9*9*97

        # spectral
        x11_b = self.conv111(X)
        x11_chunk = torch.chunk(x11_b, self.scales, 1)
        x111=x11_chunk[0]
        x112=self.conv112(x11_chunk[1])
        x113=self.conv113(x11_chunk[2]+x112)
        x114=self.conv114(x11_chunk[3]+x113)
        x11=torch.cat((x111,x112,x113,x114), dim=1)
        x11_f=self.conv115(x11)

        x12_b = self.conv121(torch.cat((X,x11_f), dim=1))
        x12_chunk = torch.chunk(x12_b, self.scales, 1)
        x121=x12_chunk[0]
        x122=self.conv122(x12_chunk[1])
        x123=self.conv123(x12_chunk[2]+x122)
        x124=self.conv124(x12_chunk[3]+x123)
        x12=torch.cat((x121,x122,x123,x124), dim=1)
        x12_f=self.conv125(x12)

        x13_b = self.conv131(torch.cat((X,x11_f,x12_f), dim=1))
        x13_chunk = torch.chunk(x13_b, self.scales, 1)
        x131=x13_chunk[0]
        x132=self.conv132(x13_chunk[1])
        x133=self.conv133(x13_chunk[2]+x132)
        x134=self.conv134(x13_chunk[3]+x133)
        x13=torch.cat((x131,x132,x133,x134), dim=1)
        x13_f=self.conv135(x13)

        x1_f=self.conv14(torch.cat((X,x11_f,x12_f,x13_f), dim=1))
        # 光谱注意力通道
        x1 = self.attention_spectral(x1_f)
        x1 = torch.mul(x1, x1_f)
        # print('X1',x1.shape)


        # spatial x
        x21_b = self.conv211(X)
        x21_chunk = torch.chunk(x21_b, self.scales, 1)
        x211=x21_chunk[0]
        x212=self.conv212(x21_chunk[1])
        x213=self.conv213(x21_chunk[2]+x212)
        x214=self.conv214(x21_chunk[3]+x213)
        x21=torch.cat((x211,x212,x213,x214), dim=1)
        x21_f=self.conv215(x21)

        x22_b = self.conv221(torch.cat((X,x21_f), dim=1))
        x22_chunk = torch.chunk(x22_b, self.scales, 1)
        x221=x22_chunk[0]
        x222=self.conv222(x22_chunk[1])
        x223=self.conv223(x22_chunk[2]+x222)
        x224=self.conv224(x22_chunk[3]+x223)
        x22=torch.cat((x221,x222,x223,x224), dim=1)
        x22_f=self.conv225(x22)

        x23_b = self.conv231(torch.cat((X,x21_f,x22_f), dim=1))
        x23_chunk = torch.chunk(x23_b, self.scales, 1)
        x231=x23_chunk[0]
        x232=self.conv232(x23_chunk[1])
        x233=self.conv233(x23_chunk[2]+x232)
        x234=self.conv234(x23_chunk[3]+x233)
        x23=torch.cat((x231,x232,x233,x234), dim=1)
        x23_f=self.conv235(x23)

        x2_f=self.conv24(torch.cat((X,x21_f,x22_f,x23_f), dim=1))
        # 空间x注意力机制 
        x2 = self.attention_spatial_x(x2_f)
        x2 = torch.mul(x2, x2_f)
        # print(x2.shape)

        # spatial y
        x31_b = self.conv311(X)
        x31_chunk = torch.chunk(x31_b, self.scales, 1)
        x311=x31_chunk[0]
        x312=self.conv312(x31_chunk[1])
        x313=self.conv313(x31_chunk[2]+x312)
        x314=self.conv314(x31_chunk[3]+x313)
        x31=torch.cat((x311,x312,x313,x314), dim=1)
        x31_f=self.conv315(x31)

        x32_b = self.conv321(torch.cat((X,x31_f), dim=1))
        x32_chunk = torch.chunk(x32_b, self.scales, 1)
        x321=x32_chunk[0]
        x322=self.conv322(x32_chunk[1])
        x323=self.conv323(x32_chunk[2]+x322)
        x324=self.conv324(x32_chunk[3]+x323)
        x32=torch.cat((x321,x322,x323,x324), dim=1)
        x32_f=self.conv325(x32)

        x33_b = self.conv331(torch.cat((X,x31_f,x32_f), dim=1))
        x33_chunk = torch.chunk(x33_b, self.scales, 1)
        x331=x33_chunk[0]
        x332=self.conv332(x33_chunk[1])
        x333=self.conv333(x33_chunk[2]+x332)
        x334=self.conv334(x33_chunk[3]+x333)
        x33=torch.cat((x331,x332,x333,x334), dim=1)
        x33_f=self.conv335(x33)

        x3_f=self.conv34(torch.cat((X,x31_f,x32_f,x33_f), dim=1))

        # 空间y注意力机制 
        x3 = self.attention_spatial_y(x3_f)
        x3 = torch.mul(x3, x3_f)
        # print(x3.shape)

        # model1
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        
        x2 = self.batch_norm_spatial_x(x2)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x3 = self.batch_norm_spatial_y(x3)
        x3= self.global_pooling(x3)
        x3 = x3.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2, x3), dim=1)

        output = self.full_connection(x_pre)

        return output












if __name__=='__main__':
    net=TBTA_dense2net(200,16).cuda()
    x=torch.randn((8,1,9,9,200)).cuda()
    y=net(x)
    print(y.shape)