# -*- coding: utf-8 -*-
"""
Flask Application
"""
#!/usr/bin/env python
# encoding: utf-8
#import dependencies
from flask import Flask, request
from klassifizierung_aus_rohdaten import klassifizierung

app = Flask(__name__)

@app.route('/', methods=['POST'])
def python_script():
    
    inputvalue = request.json
    data = inputvalue.get("data", []) 
   
    result = klassifizierung(data[0], data[1], data[2])
    return (result) 

 
    


if __name__ == '__main__':
    app.run()