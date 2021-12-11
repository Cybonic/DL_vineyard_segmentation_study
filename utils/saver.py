import os 
from datetime import datetime

class saver():
    def __init__(self,**arg):
         
        self.result_dir = arg['result_dir']
        self.name = arg['file']
        self.write_flag = 'a' # Append information to the existing file
        if not os.path.isdir(self.result_dir ):
            os.makedirs(self.result_dir )

        self.result_file = os.path.join(self.result_dir,self.name  + '.txt')
        
    def save_results(self,results_dic):
        # Convertion 
        results_str = self.dic2str(results_dic)
        self.save_to_file(results_str)
        return(results_str)
        

    def dic2str(self,results):
        now = datetime.now()
        current_time = now.strftime("%d|%H:%M:%S")
        if not os.path.isfile(self.result_file):
            self.write_flag = 'w'
        f = open(self.result_file,self.write_flag)
        
        line = "{}||".format(now)

        if isinstance(results,dict):
            for key, values in results.items():
                line += "{}:{} ".format(key,values)
        return(line)

    def save_to_file(self,line_str):
        if not isinstance(line_str,str):
            raise ValueError("Input is not a string")
        f = open(self.result_file,'a')
        f.write(line_str + '\n')
        f.close()
    
    def get_result_file(self):
        return(self.result_file)
        
    def save_model(self,model):
        return(NotImplementedError)
