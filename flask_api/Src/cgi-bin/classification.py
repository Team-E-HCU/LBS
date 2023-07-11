# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:16:07 2023

@author: lukas
"""
#!/usr/bin/env python
# encoding: utf-8
import dependencies
from flask import Flask, request
from klassifizierung_2D_ausVariable import klassifizierung

app = Flask(__name__)

@app.route('/', methods=['POST'])
def python_script():
    
    inputvalue = request.data
    result = klassifizierung(inputvalue.decode("utf-8"))
 
    return (result) 


if __name__ == '__main__':
    app.run()