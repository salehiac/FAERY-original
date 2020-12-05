# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import subprocess
import os
from datetime import datetime

import numpy as np
import functools

def get_current_time_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def bash_command(cmd:list):
    """
    cmd  list [command, arg1, arg2, ...]
    """
    #print("****************** EXECUTING *************",cmd)
    #input()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ret_code=proc.returncode

    return out, err, ret_code

def create_directory_with_pid(dir_basename,remove_if_exists=True,no_pid=False):
    while dir_basename[-1]=="/":
        dir_basename=dir_basename[:-1]
    
    dir_path=dir_basename+str(os.getpid()) if not no_pid else dir_basename
    if os.path.exists(dir_path):
        if remove_if_exists:
            bash_command(["rm",dir_path,"-rf"])
        else:
            raise Exception("directory exists but remove_if_exists is False")
    bash_command(["mkdir", dir_path])
    notif_name=dir_path+"/creation_notification.txt"
    bash_command(["touch", notif_name])
    with open(notif_name,"w") as fl:
        fl.write("created on "+get_current_time_date()+"\n")
    return dir_path

class colors:
    red=(255,0,0)
    green=(0,255,0)
    blue=(0,0,255)

if __name__=="__main__":
    _=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=True)
    dir_path=create_directory_with_pid(dir_basename="/tmp/report_1",remove_if_exists=True,no_pid=False)

