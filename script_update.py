import git # https://stackoverflow.com/questions/27400619/how-to-construct-git-diff-query-in-python
import sys 
import shutil
import argparse
import os

import pathlib


def parse_git_arg(arg):
    list_files = arg.split('M\t')
    out_list=[]
    for a in list_files:
        if a == '':
            continue 
        clean_name = a.split('\n')[0]
        out_list.append(clean_name)
    return(out_list)


def update_changed_files(dest_dir):

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    repo = git.Repo('')
    git_arg =  repo.git.diff( **{'name-status': True})
    changed_files = parse_git_arg(git_arg)
    
    print("Changed Files\n")
    [print(file) for file in changed_files]
    
    for file in changed_files:
        destination = os.path.join(dest_dir,file)
        try:
            shutil.copy(file, destination)
        except:
            print("[Error] File not copied: " + destination )

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
      '--root', '-m',
      type=str,
      required=False,
      default='/home/tiago/desktop_home/workspace',
      help='Directory to get the trained model.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    root  = FLAGS.root
    
    package_name = pathlib.Path(__file__).parent.absolute().name
    dst_dir = os.path.join(root,package_name)
    # print(package_name)

    update_changed_files(dst_dir)
 

  

  


 


