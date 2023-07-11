# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 19:32:22 2023

@author: lukas
"""

import os

for i in range(0,50,1):

    old_name = f"SensorData ({i}).txt"
    new_name = f"SensorDataKreuz ({i}).txt"
    
    if os.path.isfile(new_name):
        print("The file already exists")
    else:
        # Rename the file
        os.rename(old_name, new_name)