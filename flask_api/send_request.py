# -*- coding: utf-8 -*-
"""
API Abfrage
"""
import requests
import json

url = "https://lukas-rene.de/cgi-bin/classification.cgi"
#url = "http://127.0.0.1:5000/"
wert = open("../data/Kreuze/SensorDataKreuz (10).txt").read()

payload = {"messungen": wert}
#payload_json = json.dumps(payload)


response = requests.post(url, wert)

if response.status_code == 200:
    result = response.text
    print("Ergebnis:", result)
else:
    print("Fehler bei der Anfrage:", response.status_code)
    
    