
import os

def run_script(**arg):

    train_arg_list = []
    if "model" in arg:
        model    = arg['model']
        train_arg_list.append('--model')
        train_arg_list.append(model)

    session  = arg['session']
    cmd      = arg['cmd']

    train_arg_list.append('--session')
    train_arg_list.append(session)

    if 'results' in arg:
        value = arg['results']
        train_arg_list.append('--results')
        train_arg_list.append(value)

    if 'plot' in arg:
        value = str(arg['plot'])
        train_arg_list.append('--plot')
        train_arg_list.append(value)
    
    if 'writer' in arg:
        value = arg['writer']
        train_arg_list.append('--writer')
        train_arg_list.append(value)
    
    # Add pretrained if it exists 
    if "pretrained" in arg:
        value = arg['pretrained']
        if os.path.isfile(value + '.pth') ==  True:
            train_arg_list.append('--pretrained')
            train_arg_list.append(value)
    
    # Convert arguments to str line
    train_arg = ' '.join(train_arg_list)
    # Build Full terminal command 
    terminal_cmd_list = ['python3','-W','ignore' , cmd, train_arg]
    terminal_cmd      = ' '.join(terminal_cmd_list)
    
    print("\n\n======================================================")
    print("======================================================\n\n")

    print("[INF] $: %s\n"%(terminal_cmd))
    os.system(terminal_cmd)



import inspect

def __LINE__():
    stack_t = inspect.stack()
    ttt = inspect.getframeinfo(stack_t[1][0])
    return ttt.lineno


def __FUNC__():
    stack_t = inspect.stack()
    ttt = inspect.getframeinfo(stack_t[1][0])
    return ttt.function


def __FILE__():
    stack_t = inspect.stack()
    ttt = inspect.getframeinfo(stack_t[1][0])
    return ttt.filename