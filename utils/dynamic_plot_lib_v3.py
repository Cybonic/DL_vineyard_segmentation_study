import matplotlib
#matplotlib.use("qt4agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
#import seaborn as sns
from matplotlib.path import Path
import os

#plt.ion()
#plt.show(block=False)

'''
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

'''


class dynamic_hist():
    def __init__(self,ax,resol=1,c='g',label='',point_flag = False,save=False,plot_n=-1):
        self.point_flag = point_flag
        self.ax = ax
        self.save = save
        self.x,self.y = np.array([]),np.array([])
        self.c = c
        self.resol = resol
        self.bin_width = resol - (resol*0.1)
        self.offset = resol/2
        self.bins=np.arange(0,100+resol,resol)
        self.plot_n = plot_n
        hist, bins = np.histogram([], [],normed=True,density = True)
        self.sp = []
        self.sliding_window = 10

        
        #self.sp = self.ax.hist(hist,bins = bins, color= self.c,alpha=0.7,label=label)
        #self.sp = self.ax.bar(bins,hist)
         # Change width depending on your bins
         # Change width depending on your bins
        self.label = label

    def update_plot_elem(self,x,color=[],arg={}):
        x *=100
        nplots = arg['n_plots']
        width = self.bin_width/nplots
        bin_init_offset = self.bin_width/2
        if self.save == True:
            self.x = np.append(self.x,x)
        else: 
            self.x = x

        
        #min_v = min(self.x)
        #max_v = max(self.x)

        #self.bins = np.arange(min_v,max_v)
        hist, bins = np.histogram(self.x, self.bins,normed=True,density = True)
        
        max_val = hist.sum()
        norm_hist = hist/max_val
        if len(self.sp)==0:
            bar_bins = bins[:-1] + self.offset - bin_init_offset + (self.plot_n*width)
            self.sp = self.ax.bar(
                            bar_bins,
                            hist,
                            width=width,
                            label=self.label)
        else:
            for i in range(len(self.sp)):
                self.sp[i].set_height(hist[i])
        #hist, bins = np.histogram(self.x, self.bins,normed=True,density = True)
        #self.sp[0] = hist
        #self.sp[1] = bins
        # self.ax.hist(hist,bins = bins, color= self.c,alpha=0.7,label=self.label)

class dynamic_scatter():
    def __init__(self,ax,c='g',label='',point_flag = False,save=False,**arg):
        self.point_flag = point_flag
        self.ax = ax
        self.save = save
        self.x,self.y = np.array([]),np.array([])
        self.marker = 'o'
        self.scale = 20
        if 'marker' in arg:
           self. marker = arg['marker']
        
        if 'scale' in arg: 
            self.scale = arg['scale'] 

        self.sp = self.ax.scatter([],[],s =self.scale ,c = c,marker = self.marker, label= label)
    
    def update_plot_elem(self,x,y,color=[],arg = {}):
        
        if self.save == True:
            self.x = np.append(self.x,x)
            self.y = np.append(self.y,y)
            #self.y.append(y)
        else: 
            self.x = x
            self.y = y
        
        data = np.c_[self.x,self.y]
        self.sp.set_offsets(data)

class dynamic_plot_elm():       
    def __init__(self,ax,c='g',label='',point_flag = False ,save=False , **kwarg):
        self.point_flag = point_flag
        self.window = kwarg['window']
        self.ax    = ax
        self.color = c 
        
        linestyle = '-'
        scale = 2

        if 'scale' in kwarg:
            scale = kwarg['scale']

        if 'linestyle' in kwarg:
            linestyle = kwarg['linestyle']

        self.fill = self.ax.fill_between([],[],color=c)
        self.p, = self.ax.plot([],[],color = c,label=label,linewidth=scale,linestyle = linestyle)
    
        self.save = save
        self.x,self.y = np.array([]),np.array([])
    
    def fill_area(self,std):

        self.fill.remove()
        low_y = self.y-std
        up_y  = self.y+std
        self.fill = self.ax.fill_between(self.x,up_y,low_y,edgecolor='#1B2ACC', facecolor='#1B2ACC',alpha=0.3)
            
    def update_plot_elem(self,x,y,color=[],arg={}):
        
        if self.save == True:
            self.x = np.append(self.x,x)
            self.y = np.append(self.y,y)
        else: 
            self.x = x
            self.y = y

        df =  pd.DataFrame({'x':self.x ,'y':self.y})

        if len(self.x)>self.window and self.window >0 :
            # calculate a 60 day rolling mean and plot
            mean_pts = df.rolling(window=self.window).mean()

            mpx = mean_pts['x'][pd.isna(mean_pts['x']) == False].to_numpy()
            mpy = mean_pts['y'][pd.isna(mean_pts['y']) == False].to_numpy()

            std_pts = df.rolling(window=self.window).std()
            stdx = std_pts['x'][pd.isna(mean_pts['x']) == False].to_numpy()
            stdy = std_pts['y'][pd.isna(mean_pts['y']) == False].to_numpy()

            self.p.set_data(mpx,mpy)
            self.fill.remove()
            self.fill = self.ax.fill_between(mpx,mpy-stdy,mpy+stdy,edgecolor='#1B2ACC', facecolor='#1B2ACC',alpha=0.3)
        else:
            self.p.set_data(self.x,self.y)

        if self.point_flag == True:
            self.sp.set_offsets(np.c_[self.x,self.y])

        if color != []:
            self.p.set_color(color)

    def get_data(self):
        return(self.x, self.y)

class dynamic_plot(dynamic_plot_elm):       
    def __init__(self,title='',xlabel='',ylabel='',**arg):

        SMALL_SIZE  = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 10
        fontsize = {'xtick':BIGGER_SIZE,
                    'ytick':BIGGER_SIZE,
                    'title':BIGGER_SIZE,
                    'axis':BIGGER_SIZE,
                    'legend':BIGGER_SIZE,
                    'labels':BIGGER_SIZE,
                    'figure':BIGGER_SIZE
                    }

        if 'fontsize' in arg:
            fontsize = arg['fontsize']
        # Set fontsize
        self.set_fontsize( **fontsize)

        self.fig, self.ax = plt.subplots()
        self.title  = title
        # Set title
        self.ax.set_title(title)
        # Set label
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        self.n_plots = 0
        self.plot_v = {}
        self.xx_bag = []
        self.yy_bag = []
        self.framework= [] 
        self.keys = []

    def set_fontsize(self,**fontsize):
        #  matplotlib.rcParams.update({'font.size': arg['fontsize']['axis']})

        if 'text'in fontsize:
            plt.rc('font', size=fontsize['text'])
        if 'labels'in fontsize:
            plt.rc('axes', labelsize=fontsize['labels'])
        if 'xtick'in fontsize:
            plt.rc('xtick', labelsize=fontsize['xtick'])
        if 'ytick'in fontsize:
            plt.rc('ytick', labelsize=fontsize['ytick'])
        if 'legend'in fontsize:
            plt.rc('legend', fontsize=fontsize['legend'])
        if 'title'in fontsize:
            plt.rc('figure', titlesize=fontsize['title'])
        if 'axistitle'in fontsize:
            plt.rc('axes', titlesize=fontsize['axistitle'])

    def add_plot(self,key,color='r',point_flag = False, save = False,framework = 'line',label='', **kwarg):
        self.n_plots += 1
        self.framework.append(framework)
        self.keys.append(key)

        if framework == 'line':
            plot_elm = dynamic_plot_elm(self.ax,c=color,point_flag=point_flag,save=save,label=label,**kwarg)
        elif framework == 'scatter':
            plot_elm = dynamic_scatter(self.ax,c=color,point_flag=point_flag,save=save,label=label,**kwarg)
        elif framework == 'hist':
            plot_elm = dynamic_hist(self.ax,c=color,label=label,point_flag=point_flag,save=save,plot_n=self.n_plots)
        self.plot_v[key]=plot_elm
        

    def update_plot(self,key,x,y,color=[]):
        if self.framework == 'hist':
            self.plot_v[key].update_plot_elem(x,y,arg={'n_plots':self.n_plots})
        else:
            self.plot_v[key].update_plot_elem(x,y)

        self.xx_bag = np.append(self.xx_bag,x)
        self.yy_bag = np.append(self.yy_bag,y)

    #def save_data(self,name):
    def addon(self,key,**arg):
        if 'fill' in arg:
            self.plot_v[key].fill_area(arg['fill'])

    def axis_adjust(self,offset=0,**arg):

        x = self.xx_bag 
        y = self.yy_bag
    
        if 'axis' in arg:
            axis = arg['axis']
            self.ax.axis([axis['xmin'], axis['xmax'], axis['ymin'],axis['ymax']])
        else:
            #if len(x)<zoom:
            if len(x) > 0 and len(y) > 0:
                xmax,xmin= max(x),min(x)
                ymax,ymin = max(y),min(y)
                self.ax.axis([xmin - offset, xmax + offset, ymin - offset, ymax + offset])
            #else: 
            #    xmax,xmin= max(x[-zoom:]),min(x[-zoom:])
            #    ymax,ymin = max(y[-zoom:]),min(y[-zoom:])

        
    def get_data(self,label):
        return(self.plot_v[label].get_data())

    def save_data_file(self,root):
        if not os.path.isdir(root):
            os.makedirs(root)
        file = os.path.join(root,self.title)
        print("[INF] File: " + file)
        
        with open(file,'w') as f: 
            for key, value in self.plot_v.items():
                x,y = value.get_data()
                
                str_x = key + ':x:' + ' '.join([str(round(v,3)) for v in x])
                str_y = key + ':y:' + ' '.join([str(round(v,3)) for v in y])
                
                f.write(str_x + '\n')
                f.write(str_y + '\n')
                print("[INF] Saved Label: " + key)
            #file = os.path.join(path,label)
            
            
    def show(self,time=0.0001,offset=0.1,**arg):
        self.ax.legend()
        fontsize=16
        if 'axis' in arg:
            axis = arg['axis']
            self.axis_adjust(offset=offset,axis=axis)
        else:
            self.axis_adjust(offset=offset)
        
        self.fig.canvas.draw()
        if 'grid_on' in arg:
            grid_on = arg['grid_on']
            if grid_on == True:
                self.ax.grid(True)

        #plt.draw()
        #plt.pause(0.001)
        plt.pause(time)
    
    def hold_fig(self):
        plt.show()