from flask import Flask, request, make_response, jsonify, send_file
from flask_restful import Api, Resource
import requests
import random
import time
import json
import os

class ScannerApi(Resource):
    
    def get(self):
        data = request.get_json()
    
    
    def put(self):
        data = request.get_json()
    
    # get signal to start scanning
    def post(self):
        data = request.get_json()
        if data["start_scanning"]:
            barcode = random.choice(os.listdir('emul_results'))
            n = len(os.listdir(f'emul_results\\{barcode}'))
            for idx, i in enumerate(os.listdir(f'emul_results\\{barcode}')):
                print(f"Sending result (barcode: {barcode}) [{idx}/{n}]")
                files = [
                    ('image', (i, open(f'{os.path.abspath(os.getcwd())}\\emul_results\\{barcode}\\{i}', 'rb'), 'image/png')),
                    ('data', ('datas', json.dumps({'id': i, 'token': "a27d94e52cfd2af22a0b4232648ad92f7a4a29ee683e6013b348c133605adb1e", 'barcode': barcode, 'scanner_id': data["scanner_id"], 'is_first': idx == 0, 'is_last': idx == n - 1}), 'application/json')),
                ]
                response = requests.put("http://127.0.0.1:7070/api/v1/result", files=files)
                # time.sleep(3)
                
                
"""
curl -H "Content-Type: application/json" -X POST 127.0.0.1:8000/api/v1/scanner -d "{\"start_scanning\": 1}"
"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sHjla91HdhnBS29QDlshPEUDqG29XzTmDUBO3tSjmHFaKQQwGKl0vD3m1rFD4WV0Y7a7CTJT5wrHIVQF915da6GDHB4qxhbJ9clt4vxlIOu0x0I87bLeB0dyCWQD1iO6S'

api = Api(app, prefix='/api/v1')
api.add_resource(ScannerApi, '/scanner')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)