#!/usr/bin/python3
import os

if os.environ['REQUEST_METHOD'] == 'OPTIONS':
    # Setze die erforderlichen CORS-Header für die OPTIONS-Anfrage
    print("Access-Control-Allow-Origin: *")
    print("Access-Control-Allow-Methods: POST")
    print("Access-Control-Allow-Headers: Content-Type")
    print("Content-Length: 0")
    print("Content-Type: text/plain")
    print()
    exit()

try:
    from wsgiref.handlers import CGIHandler
    from classification import app

    CGIHandler().run(app)
    
except Exception as err:
    print("Content-Type: text/html\n")
    print(err)
