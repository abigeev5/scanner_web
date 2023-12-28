from flask import request, Blueprint, redirect, url_for
import requests

from config import server_api, session, error_codes

blueprint = Blueprint('utils_page', __name__, template_folder='templates')

def user_wrapper(f):
    def wrapper(*args):
        global session
        if request.cookies.get('token'):
            token = token if token else request.cookies.get('token')
            user = get_user_info(token)
            return f(user, token, *args)
        else:
            return redirect(url_for('main_pages.home'))
    return wrapper


def get_user_info(token):
    global session
    response = session.get(server_api + '/user', json={'token': token})
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': response.status_code}


@blueprint.route('/delete_img')
def delete_img():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        # user = get_user_info(token)
        data = dict(request.args)
        response = session.delete(server_api + '/result', json={'token': token, 'barcode': data['barcode'], 'filename': data['filename']})
        return {'code': response.status_code, "message": error_codes.get(response.status_code, '')}
    return {'code': "-1", "message": ""}


@blueprint.route('/start_scanning')
def start_scanning():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        data = dict(request.args)
        response = requests.post("http://127.0.0.1:8000/api/v1/scanner", json={"start_scanning": 1, "scanner_id": data['scanner_id']})
        return {'code': response.status_code, "message": error_codes.get(response.status_code, '')}
    return {'code': "-1", "message": ""}


@blueprint.route('/update_user')
def update_user():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        data = dict(request.args)
        data["token"] = token
        if data["create"] == "true":
            response = session.put(server_api + '/user', json=data)
        else:
            response = session.post(server_api + '/user', json=data)
        for scanner in session.get(server_api + "/scanner", json={"token": token}).json()["response"]:
            operators = scanner["operators"]["operators"]
            print(data)
            if not(str(scanner["id"]) in data["scanners"].split(", ")):
                if data["id"] in operators:
                    operators.remove(data["id"])
            else:
                if not(data["id"] in operators):
                    operators.append(data["id"])
            session.post(server_api + "/scanner", json={"token": token, "id": scanner["id"], "operators": {"operators": operators}})
        return {'code': response.status_code, "message": error_codes.get(response.status_code, '')}
    return {'code': "-1", "message": ""}