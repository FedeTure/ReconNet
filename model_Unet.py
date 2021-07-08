import torch
import torch.nn as nn

class ReconNet(nn.Module):
    
    def initial_conv(self, in_channels, out_channels):        
        return nn.Sequential(nn.Conv3d(in_channels, int(out_channels/2), 3, padding = 1),
                nn.BatchNorm3d(int(out_channels/2)),
                nn.ReLU(inplace=True),
                nn.Conv3d(int(out_channels/2), out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv(self, in_channels, out_channels):         
        return nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding = 1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding = 1), # HEEEERE IT WASS IN OUT
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding = 1), #HAND HERE IN IN
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
                
    def __init__(self,num_channels,num_latents):
        super(ReconNet, self).__init__()
        
        print('num_channel',num_channels)
        print('num_latent',num_latents)
        self.num_channels=num_channels      
        self.conv_initial = self.initial_conv(1, num_channels)  
        self.conv_final = nn.Conv3d(num_channels, 1, 3, padding = 1)  
        
        self.conv_rest_x_64 = self.consecutive_conv(num_channels, num_channels*2) 
        self.conv_rest_x_32 = self.consecutive_conv(num_channels*2, num_channels*4) 
        self.conv_rest_x_16 = self.consecutive_conv(num_channels*4, num_channels*8) 

        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels*8+num_channels*4, num_channels*4) 
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels*4+num_channels*2, num_channels*2) 
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels*2+num_channels, num_channels) 
        
        #self.linear_enc=nn.Linear(16*16*16*num_channels*8,num_latents)
        #self.linear_dec=nn.Linear(num_latents,16*16*16*num_channels*8)
        
        self.contract = nn.MaxPool3d(2, stride=2) 
        self.expand = nn.Upsample(scale_factor=2)

            
    def forward(self,x):
        x_128 = self.conv_initial(x) #conv_initial 1->16->32
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64) #rest 32->32->64
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32) #rest 64->64->128
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16) #rest 128->128->256

        #x_flat=x_16.view(-1,16*16*16*self.num_channels*8) # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        
        #fc=self.linear_enc(x_flat)
        #u_16 = self.linear_dec(fc).view(-1,self.num_channels*8,16,16,16)

        u_32 = self.expand(x_16)
        u_32 = self.conv_rest_u_32(torch.cat((x_32, u_32),1)) #rest 256+128-> 128 -> 128
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(torch.cat((x_64,u_64),1))  #rest 128+64-> 64 -> 64
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(torch.cat((x_128, u_128),1))  # rest 64+32-> 32 -> 32
        u_128 = self.conv_final(u_128)

        S = torch.sigmoid(u_128)
        
        return S



   
class Discriminator(nn.Module):
    def __init__(self, num_channels, num_features):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (num_channels) x 128 x 128
            nn.Conv3d(num_channels, num_features, 4, 2, 1, bias=False), # kernel 4, stride 2, padding 1 -> pooling not used 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features) x 64 x 64
            
            nn.Conv3d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*2) x 32 x 32
            
            nn.Conv3d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*4) x 16 x 16
            
            nn.Conv3d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (num_features*8) x 8 x 8
             
            nn.Conv3d(num_features * 8, num_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*16) x 4 x 4
            
            nn.Conv3d(num_features * 16, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
    
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
    
    
class Discriminator_new(nn.Module):
    def __init__(self, num_channels, num_features):
        super(Discriminator_new, self).__init__()
        self.main = nn.Sequential(
            # input is (num_channels) x 128 x 128 x128
            nn.Conv3d(num_channels, num_features, 4, 2, 1, bias=False), # kernel 4, stride 2, padding 1 -> pooling not used 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features) x 64 x 64 x 64
            
            nn.Conv3d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*2) x 32 x 32 x 32
            
            nn.Conv3d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*4) x 16 x 16 x 16
            
            nn.Conv3d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
             # state size. (num_features*8) x 8 x 8 x 8
             
            nn.Conv3d(num_features * 8, num_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm3d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*16) x 4 x 4 x 4
            
            Flatten(),
            
            nn.Linear(num_features * 16 * 4**3, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 1)
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)