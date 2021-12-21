import sys 
import os

from utils import utils 
from utils.scrip_utils import run_script
import platform



#TRAIN = {'t1':['esac1','esac2'],
#         't2':['esac1','valdoeiro'],
#         't3':['valdoeiro','esac2']
#        }

TRAIN = {'t1':['esac','valdoeiro'],
         't2':['qtabaixo','valdoeiro'],
         't3':['qtabaixo','esac']
        }

TEST = {'t1':['qtabaixo'],
        't2':['esac'],
        't3':['valdoeiro']
        }

ms_files = [
            #'all',
            'rgb',
            #'re',
            'nir'
            #'thermal',
            #'rgb_thermal',
            #'rgb_thermal_re',
            #'rgb_thermal_nir',
            #'thermal_re_nir',
            #'thermal_re',
            #'thermal_nir',
            #'rgb_re_nir',
            #'rgb_re',
            #'rgb_nir',
            #'re_nir'
        ]

hd_files = 'rgb'

LR = {'segnet':0.001,'unet_bn':0.001,'modsegnet':0.001}
WD = {'segnet':0.000001,'unet_bn':0.000001,'modsegnet':0.00001}

REPETITIONS = 1

EXEC_FILE = 'train.py'
CMD = 'python3'
PLOT_FLAG = 0




TEST_PARAMETERS = {
'LEARNING_RATE': 0.0001,
'WEIGHT_DECAY':  0.00004,
'BATCH_SIZE':  1,
'SHUFFLE':  False,
'MAX_EPOCH': 1,
'VAL_EPOCH': 1,
'amsgrad': False,
'TEMP_SESSION': 'temp',
'AUGMENT': True,
'FRACTION': 0.01,
'USE_PRETRAINED': False,
'SAVE_PRETRAINED':True
}

DEPLOY_PARAMETERS = {
'LEARNING_RATE': 0.0001,
'WEIGHT_DECAY':  0.00001,
'BATCH_SIZE':  5,
'SHUFFLE':  True,
'MAX_EPOCH': 50,
'VAL_EPOCH': 1,
'amsgrad': True,
'TEMP_SESSION': 'temp',
'AUGMENT': True,
'FRACTION': 1,
'USE_PRETRAINED': False,
'SAVE_PRETRAINED': True
}


def UpdateSession(sessionfilepath,**arg):

    print(arg)
    temp_path = os.path.join('session',arg['TEMP_SESSION'] + '.yaml')
    t = arg['cross_val']
    
    network = arg['network']
    train = TRAIN[t]
    test = TEST[t]

    # Remove temp file 
    if os.path.isfile(temp_path):
        os.remove(temp_path)
    # Load parameters from original session file 
    
    session_settings = utils.load_config(os.path.join('session', sessionfilepath + '.yaml'))
    # ===============================================================================
    # Update parameters 
    session_settings['network']['model'] = network
    session_settings['network']['index']['NDVI'] = False
    session_settings['network']['pretrained']['use'] = arg['USE_PRETRAINED']
    session_settings['network']['pretrained']['save'] = arg['SAVE_PRETRAINED']
    session_settings['network']['pretrained']['path'] = os.path.join(sessionfilepath,t)
    session_settings['max_epochs']= arg['MAX_EPOCH']
    session_settings['report_val']= arg['VAL_EPOCH'] 
    session_settings['optimizer']['lr']= arg['LEARNING_RATE']
    session_settings['optimizer']['w_decay']= arg['WEIGHT_DECAY']
    session_settings['optimizer']['amsgrad']= arg['amsgrad']  
    session_settings['dataset']['loader']['shuffle'] = arg['SHUFFLE']
    session_settings['dataset']['loader']['batch_size'] = arg['BATCH_SIZE']  
    session_settings['dataset']['augment']= arg['AUGMENT'] 
    session_settings['dataset']['train']= train
    session_settings['dataset']['test']= test
    session_settings['dataset']['name'] = arg['SENSOR']
    session_settings['dataset']['fraction']= arg['FRACTION']
    session_settings['saver']['file'] = session_settings['network']['model'] + '_' + t
    session_settings['saver']['result_dir']= os.path.join('results',arg['RUN'])
    session_settings['run'] = arg['RUN']


    sesson_name = '_'.join(sessionfilepath.split(os.sep))
    name = '_'.join([sesson_name,t,network,
                                    'f:' +str(arg['FRACTION']),
                                    'a:' +str(int(arg['AUGMENT'])),
                                    'lr:'+str(arg['LEARNING_RATE']),
                                    'wd:'+str(arg['WEIGHT_DECAY'])
                                    ])
    # ===============================================================================
    # Save the new settings to temp file
    utils.save_config(temp_path,session_settings)
    # Run script 
    return(arg['TEMP_SESSION'] ,name)

if __name__ == '__main__':
    

    RUN = 'paper_____'
    
    pc_name = platform.node()
    print("[INFO][SCRIPT] "+ pc_name)

    LR_RANGE= [0.000001,0.00001,0.0001,0.001,0.01]
    WD_RANGE= [0.000001,0.00001,0.0001,0.001,0.01]

    if pc_name == 'tiago-lp': # Laptop (Testing)
        parameters = TEST_PARAMETERS
        print("[INFO][SCRIPT] LOADING Test Paramters")
    else: #pc_name == 'DESKTOP-5V7R599': # Desktop (deploy)
        parameters = DEPLOY_PARAMETERS 
        print("[INFO][SCRIPT] LOADING Deploy Paramters")

    networks = ['segnet','unet_bn','modsegnet']
    #networks = ['segnet']
    tx = ['t1','t2','t3']

    
    t = 't1' 
    
    for network in networks:    
            for i in range(10): #for lr in LR_RANGE:
                #for wd in WD_RANGE:
                    #print("[INFO][SCRIPT] Cycle: %f"%(lr))
                    #parameters['LEARNING_RATE'] = lr
                    #parameters['WEIGHT_DECAY'] = wd
                    session_root = 'ms'
                    parameters['SENSOR'] = 'altum' 
                    parameters['RUN'] = os.path.join(RUN,session_root)
                    for file in ms_files:
                        print("[INFO][SCRIPT] File: " + file)
                        org_session = os.path.join(session_root,file)
                        session,name = UpdateSession(org_session, cross_val = t, network = network, **parameters)
                        name = name + '_i:' + str(i)  
                        run_script(session = session, cmd = EXEC_FILE,writer = name)
                


    #for i in range(1):
    #    for network in networks:    
    #    #for t in tx:
    #            parameters['LEARNING_RATE'] = LR[network]
    #            parameters['WEIGHT_DECAY'] = WD[network]
    #            session_root = 'hd'
    #            parameters['RUN'] = os.path.join(RUN,session_root)
    #            parameters['SENSOR'] = 'x7' 
    #            org_session = os.path.join(session_root,hd_files)
    #            session,name = UpdateSession(org_session, cross_val = t, network = network, **parameters) 
    #            name = name + '_i:' + str(i) 
    #            run_script(session = session, cmd = EXEC_FILE,writer = name)
