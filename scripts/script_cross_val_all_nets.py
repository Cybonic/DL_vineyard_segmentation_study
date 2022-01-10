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
            'all',
            'rgb',
            're',
            'nir'
            'thermal',
            'rgb_thermal',
            'rgb_thermal_re',
            'rgb_thermal_nir',
            'thermal_re_nir',
            'thermal_re',
            'thermal_nir',
            'rgb_re_nir',
            'rgb_re',
            'rgb_nir',
            're_nir'
        ]

hd_files = 'rgb'

LR = {'segnet':0.000171,'unet_bn':0.0001,'modsegnet':0.00011029}
WD = {'segnet':0.000061268,'unet_bn':0.00004,'modsegnet':0.0006758}

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
'FRACTION': 1,
}

DEPLOY_PARAMETERS = {
'LEARNING_RATE': 0.0001,
'WEIGHT_DECAY':  0.00004,
'BATCH_SIZE':  10,
'SHUFFLE':  True,
'MAX_EPOCH': 100,
'VAL_EPOCH': 1,
'amsgrad': True,
'TEMP_SESSION': 'temp',
'AUGMENT': True,
'FRACTION': 1,
'SET': 'T1'
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
    session_settings['network']['pretrained']['use'] = True
    session_settings['max_epochs']= arg['MAX_EPOCH']
    session_settings['report_val']= arg['VAL_EPOCH'] 
    session_settings['optimizer']['lr']= LR[network]
    session_settings['optimizer']['w_decay']= WD[network]
    session_settings['optimizer']['amsgrad']= arg['amsgrad']  
    session_settings['dataset']['loader']['shuffle'] = arg['SHUFFLE']
    session_settings['dataset']['loader']['batch_size'] = arg['BATCH_SIZE']  
    session_settings['dataset']['augment']= arg['AUGMENT'] 
    session_settings['dataset']['train']= train
    session_settings['dataset']['test']= test
    session_settings['dataset']['fraction']= arg['FRACTION']
    session_settings['saver']['file'] = session_settings['network']['model'] + '_' + t
    session_settings['saver']['result_dir']= 'results/robustness' 
    # ===============================================================================
    # Save the new settings to temp file
    utils.save_config(temp_path,session_settings)
    # Run script 
    return(arg['TEMP_SESSION'] )

if __name__ == '__main__':

    
    pc_name = platform.node()
    print("[INFO][SCRIPT] "+ pc_name)


    if pc_name == 'tiago-lp': # Laptop (Testing)
        parameters = TEST_PARAMETERS
        print("[INFO][SCRIPT] LOADING Test Paramters")
    else: #pc_name == 'DESKTOP-5V7R599': # Desktop (deploy)
        parameters = DEPLOY_PARAMETERS 
        print("[INFO][SCRIPT] LOADING Deploy Paramters")

    networks = ['segnet','unet_bn','modsegnet']
    tx = ['t1','t2','t3']

    for cnt in range(REPETITIONS):
        print("[INFO][SCRIPT] Cycle: %d"%(cnt))
        for network in networks:    
            for t in tx:
                #ms_session_root = 'ms'
                
                #for file in ms_files:
                #    print("[INFO][SCRIPT] File: " + file)
                #    org_session = os.path.join(ms_session_root,file)
                #    session = UpdateSession(org_session, cross_val = t, network = network, **parameters) 
                #    run_script(session = session, cmd = EXEC_FILE,plot = PLOT_FLAG)
                print("[SCRIPT] Test: " + t)
                name = '_'.join(['hd',t,network,
                                    'f',str(parameters['FRACTION']),
                                    'a',str(parameters['AUGMENT']),
                                    'lr',str(parameters['LEARNING_RATE']),
                                    'wd',str(parameters['WEIGHT_DECAY'])
                                    ])
                hd_session_root =  'hd'
                org_session = os.path.join(hd_session_root,hd_files)
                session = UpdateSession(org_session, cross_val = t, network = network, **parameters) 
                run_script(session = session, cmd = EXEC_FILE,writer = name)
