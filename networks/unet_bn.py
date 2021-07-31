# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:55:14 2021

@author: Pedro Conde
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

#O model que se segue é uma UNET com algumas adaptações relativamente ao
#modelo original, todas devidamente referidas.


# Definimos a dupla convolução usada várias vezes no modelo.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,drop_rate=0):
        super(DoubleConv, self).__init__()
        layers = [
            #Ao contrário do modelo original, as convoluçõs têm padding=1.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #Ao contrário do modelo original, utilizaremos batch normalization.
            #Por essa razão o parâmetro "bias" é "False" na convolução.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if drop_rate > 0:
            layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)
    def forward(self, x):
        return self.features(x)

class UNET(nn.Module):
    #Os features representam o número de canais resultantes 
    #após cada convolução.
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], drop_rate = 0.5
    ):
        super(UNET, self).__init__()
        #Criamos duas listas vazias onde colocaremos as convoluções duplas
        #para cada parte da rede
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Colocamos as convoluções, para a parte descendente, 
        #na lista correspondente.
        self.encoder1 =  DoubleConv(in_channels, features[0])
        self.encoder2 =  DoubleConv(features[0], features[1])
        self.encoder3 =  DoubleConv(features[1], features[2])
        self.encoder4 =  DoubleConv(features[2], features[3])

        for i,feature in enumerate(features):
            if i > 2:
                self.downs.append(DoubleConv(in_channels, feature,drop_rate=drop_rate))
            else:
                self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Colocamos as convoluções, para a parte ascendente, 
        #na lista correspondente.
        for feature in reversed(features):
            #Primeiro é colocada uma convolução transposta, 
            #que irá reduzir o número de canais.
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            #A convolução transposta é intercalada com a convolução dupla.
            self.ups.append(DoubleConv(feature*2, feature, drop_rate = drop_rate))
            
        #O bottleneck é referente à ultima dupla convolução da parte
        #descendente, antes de iniciar a parte ascendente.
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        #Define-se finalmente a ultima convolução, que dará o output.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Comecemos por criar um lista vazia, que irá acumular os outputs da
        #parte descendente da rede, para as concatenções da parte ascendente.
        skip_connections = []
        
        #Parte descendente da rede:
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        
        #Invertemos agora a lista para aplicar na parte ascendente.
        skip_connections = skip_connections[::-1]
        
        #Parte ascendente da rede:
        for idx in range(0, len(self.ups), 2):
            #Começamos com a convolução tranposta, situada numa posição par
            #da lista.
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            #O paço seguinte será utilizado no situação em que o input
            #não é divisível por 16.
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            #Aplicamos então a dupla convolução, situada numa posição ímpar
            #da lista.
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


#Teste:
def test():
    x = torch.randn((5, 6, 160, 160))
    model = UNET(in_channels=6, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    #assert preds.shape == x.shape

if __name__ == "__main__":
    test()
