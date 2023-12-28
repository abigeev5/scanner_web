from flask import request, make_response, jsonify, send_file
from flask_restful import Resource
from utils import get_user, make_error, make_success, verify_token, create_token
from models import User, Result, Scanner, Version
import db_session
from datetime import datetime
import json
import cv2
import os

from egg_detection.inference import inference

class AuthApi(Resource):
    
    # get token by login and password
    # input: username, hashed password
    # output: token
    @verify_token(required_args=['username', 'password'])
    def get(self, user, data):
        user = get_user(username=data["username"])
        if user.password == data['password']:
            return make_response(jsonify({'token': user.token}), 200)
        else:
            return make_error(401)


class UserApi(Resource):
    # get info
    # input: token
    # output: username, email, role
    @verify_token(required_args=['token'])
    def get(self, user, data):
        # print("[DEBUG] UserApi GET:", user)
        session = db_session.create_session()
        if ("get_all" in data) and (user.role == 2):
            users = session.query(User).all()
            response = list()
            for cur_user in users:
                scanners = [i.id for i in session.query(Scanner).filter(Scanner.operators.contains(f'"{cur_user.id}"')).all()]
                response.append({
                    'id': cur_user.id,
                    'username': cur_user.username, 
                    'info': cur_user.info,
                    'name': cur_user.name, 
                    'role': cur_user.role,
                    'scanners': scanners})
            session.close()
            return ({'response': response}, 200)
        else:
            scanners = [i.id for i in session.query(Scanner).filter(Scanner.operators.contains(f'"{user.id}"')).all()]
            session.close()
            return (user.to_json(), 200)
            

    # Create user
    # input: username, password, name, role, token (admin)
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def put(self, admin, data):
        session = db_session.create_session()
        user = session.query(User).filter(User.id == data["id"]).first()
        if user == None:
            user = User()
            user.from_json(data)
            user.token = create_token()
            user.role = 1

            session.add(user)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(409)


    # Update user
    # input: id, role, token (admin)
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def post(self, admin, data):
        session = db_session.create_session()
        user = session.query(User).filter(User.id == data["id"]).first()
        if user:
            user.from_json(data)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)


    # Delete user
    # input: username, token (admin)
    # output: result code
    @verify_token(role=2, required_args=['token', 'username'])
    def delete(self, admin, data):
        user = get_user(username=data['username'])
        if user:
            session = db_session.create_session()
            session.delete(user)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)


class ResultApi(Resource):
    
    # get results
    # input: scanner_id
    # output (if scanner): {results: {barcode: [list of dicts with metadata]]}
    @verify_token(required_args=['token', 'scanner_id'])
    def get(self, user, data):
        session = db_session.create_session()
        results = session.query(Result).filter(Result.scanner_id == data['scanner_id']).all()
        
        users = {user_.id: user_.username for user_ in session.query(User).all()}
        scanners = {scanner.id: scanner.name for scanner in session.query(Scanner).all()}
        
        response = {'results': {}}
        for result in results:
            response['results'][result.barcode] = result.results
            response['results'][result.barcode].update({'user': users[result.user_id], 'scanner': scanners[result.scanner_id], 'enterobiasis': result.enterobiasis})
        session.close()
        return (response, 200)


    # store images
    # input: barcode, scanner_id, pairs {id: filename}
    # output: result code
    # @verify_token(required_args=['token', 'barcode', 'scanner_id'])
    #def put(self, user, data):
    def put(self):
        data = json.load(request.files['data'])
        response = ""
        if 'image' in request.files:
            file = request.files['image']
            if data["is_first"]:
                os.mkdir(os.path.join('results', data['barcode']))
            if file:
                file_name = f"{datetime.now().microsecond}"
                file.save(os.path.join('results', data['barcode'], f"{file_name}.png"))
                image = cv2.imread(os.path.join('results', data['barcode'], f"{file_name}.png"))
                cv2.imwrite(os.path.join('results', data['barcode'], f"thumb_{file_name}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if data["is_last"]:
                result_path = inference(os.path.join('results', data['barcode']))
                with open(result_path, 'r') as f:
                    data_ = json.load(f)
                results = {
                    "results": {
                        image["id"]: {
                            "image": image["file_name"], 
                            "bboxes": [], 
                            "scores": [], 
                            "thumbnail": f"thumb_{image['file_name'].replace('.png', '.jpg')}", 
                            "time": "13.10.22 15:15:15"
                        } for image in data_["images"] if not(".jpg" in image["file_name"])
                    }
                }
                for result in data_["annotations"]:
                    if result["image_id"] in results:
                        results["results"][result["image_id"]]["bboxes"].append(result["bbox"])
                        results["results"][result["image_id"]]["scores"].append(result["score"])
                results = {"results": [v for v in results["results"].values()]}
                session = db_session.create_session()
                
                res = Result()
                res.user_id = "1"
                res.scanner_id = data["scanner_id"]
                res.barcode = data["barcode"]
                res.date = "1"
                res.results = results
                res.enterobiasis = 0
                res.folder_path = f"results/{data['barcode']}"
                
                session.add(res)
                session.commit()
                session.close()
        return (response, 200)


    # update images
    # input: barcode, scanner_id, pairs {id: filename}
    # output: result code
    @verify_token
    def post(self, user, data):
        image = request.files['image']
        return ({"meaning_of_life": 42}, 200)


    # delete images
    # input: barcode, filename.png (or not to delete result)
    # output: result code
    @verify_token(required_args=['token', 'barcode'])
    def delete(self, user, data):
        session = db_session.create_session()
        if ('filename' in data):
            results = session.query(Result).filter(Result.barcode == data['barcode']).first()
            os.remove(results.folder_path + '/' + data['filename'])
            os.remove(results.folder_path + '/thumb_' + data['filename'].replace('.png', '.jpg'))
        else:
            results = session.query(Result).filter(Result.barcode == data['barcode']).all()
            session.delete(results)
            session.commit()
        session.close()
        return make_success(200)


class ScannerApi(Resource):
    
    # get list scanners and settings which is avilable for user
    # input: None
    # output: name, settings, barcodes, version, info
    @verify_token(required_args=['token'])
    def get(self, user, data):
        session = db_session.create_session()
        if user.role == 2:
            scanners = session.query(Scanner).all()
        else:
            scanners = session.query(Scanner).filter(Scanner.operators.contains(f"\"{user.id}\"")).all()
        response = [i.to_json() for i in scanners]
        session.close()
        return ({'response': response}, 200)
    
    # add scanner
    # input: name, settings, version, info, notifications
    # output: result code
    @verify_token(role=2, required_args=['token', 'name'])
    def put(self, user, data):
        scanner = Scanner()
        scanner.from_json(data)

        session = db_session.create_session()
        session.add(scanner)
        session.commit()
        session.close()
        return make_success(200)
    
    # edit scanner
    # input: id, name, settigns, version, info, notifications
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def post(self, user, data):
        session = db_session.create_session()
        scanner = session.query(Scanner).filter(Scanner.id == data["id"]).first()
        if scanner:
            scanner.from_json(data)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)
    
    
    # delete scanner
    # input: id
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def delete(self, user, data):
        scanner = session.query(Scanner).filter(Scanner.id == data["id"]).first()
        if scanner:
            session = db_session.create_session()
            session.delete(scanner)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)


class StatusApi(Resource):
    # Server status

    # input: None
    # output: time, unit test results, status
    def get(self):
        return ({
            "time": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            }, 200)


class VersionApi(Resource):
    
    # Get setting of version
    # input: None
    # output: settings
    @verify_token(required_args=['token'])
    def get(self, user, data):
        session = db_session.create_session()
        versions = session.query(Version).all()
        response = [i.to_json() for i in versions]
        session.close()
        return ({'response': response}, 200)
    
    
    # Add new version with settings
    # input: settings, name
    # output: result code
    @verify_token(role=2, required_args=['token', 'name'])
    def put(self, user, data):
        version = session.query(Version).filter(Version.name == data["name"]).first()
        if version:
            version = Version()
            version.from_json(data)

            session = db_session.create_session()
            session.add(version)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_success(409)
    
    
    # Update version settings
    # input: id, settings
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def post(self, user, data):
        session = db_session.create_session()
        version = session.query(Version).filter(Version.id == data["id"]).first()
        if version:
            version.from_json(data)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)
    
    
    # Delete version
    # input: version
    # output: result code
    @verify_token(role=2, required_args=['token', 'id'])
    def delete(self, user, data):
        version = session.query(Version).filter(Version.id == data["id"]).first()
        if version:
            session = db_session.create_session()
            session.delete(version)
            session.commit()
            session.close()
            return make_success(200)
        else:
            return make_error(404)
        