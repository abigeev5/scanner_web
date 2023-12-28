from flask import render_template, request, redirect, url_for, make_response, Blueprint
from datetime import datetime

from config import server_api, session, error_codes
from utils import get_user_info, user_wrapper


blueprint = Blueprint('main_pages', __name__, template_folder='templates')


@blueprint.route('/')
@blueprint.route('/home')
def home(user=None, token=None):
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 1:
            return redirect(url_for('main_pages.results_page'))
        else:
            return redirect(url_for('main_pages.scanners_page'))
    else:
        return redirect(url_for('login_page.login'))


@blueprint.route('/home/results/')
def results_page(token=None):
    global session
    if request.cookies.get('token') or token:
        token = token if token else request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 1:
            response = session.get(server_api + '/scanner', json={'token': token})
            data = list()
            error_code = None
            message = None
            if response.status_code == 200:
                data = response.json()['response']
                for (idx, scanner) in enumerate(response.json()['response']):
                    r = session.get(server_api + '/result', json={'token': token, 'scanner_id': scanner['id']})
                    if r.status_code == 200:
                        results = r.json()['results']
                        data[idx]['results'] = dict()
                        for (barcode, info) in results.items():
                            for i in range(len(info['results'])):
                                info['results'][i].update({
                                    'image': f"{server_api}/image/{barcode}/{info['results'][i]['image']}", 
                                    'thumbnail': f"{server_api}/image/{barcode}/{info['results'][i]['thumbnail']}",
                                    'user': info['user'],
                                    'scanner': info['scanner'],
                                    })
                            tmp = info['enterobiasis']
                            info = info['results']
                            #info['enterobiasis'] = tmp
                            if not(info[0]['time'].split()[0]) in data[idx]['results']:
                                data[idx]['results'][info[0]['time'].split()[0]] = {barcode: {'enterobiasis': tmp, 'info': info}}
                            else:
                                data[idx]['results'][info[0]['time'].split()[0]][barcode] = {'enterobiasis': tmp, 'info': info}
                    else:
                        error_code = r.status_code
            else:
                error_code = response.status_code
            return render_template('results.html', results=data, active='results', user=user, error=error_code, message=error_codes.get(response.status_code, ''))
        else:
            return redirect(url_for('main_pages.scanners_page')) 
    return redirect(url_for('main_pages.home'))


@blueprint.route('/home/scanners/')
def scanners_page():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 2:
            response = session.get(server_api + '/scanner', json={'token': token})
            versions = session.get(server_api + '/version', json={'token': token}).json()["response"]
            scanners = []
            error_code = None
            message = None
            if response.status_code == 200:
                scanners = response.json()['response']
            else:
                error_code = response.status_code
            versions = {i["name"]: i for i in versions}
            return render_template('scanners.html', active='scanners', scanners=scanners, user=user, versions=versions, error=error_code, message=error_codes.get(response.status_code, ''))
        else:
           return redirect(url_for('main_pages.results_page')) 
    return redirect(url_for('login_page.login'))


@blueprint.route('/home/versions/')
def versions_page():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 2:
            response = session.get(server_api + '/version', json={'token': token})
            data = []
            error_code = None
            message = None
            if response.status_code == 200:
                data = response.json()['response']
            else:
                error_code = response.status_code
            return render_template('versions.html', active='versions', versions=data, user=user, error=error_code, message=error_codes.get(response.status_code, ''))
        else:
           return redirect(url_for('main_pages.results_page')) 
    return redirect(url_for('login_page.login'))




@blueprint.route('/home/users/')
def users_page(user_id=0):
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 2:
            response = session.get(server_api + '/user', json={'token': token, 'get_all': True})
            users = {'Администраторы': list(), 'Операторы': list()}
            scanners = []
            error_code = None
            message = None
            if response.status_code == 200:
                users = {'Администраторы': [i for i in response.json()['response'] if i["role"] == 2], 
                         'Операторы': [i for i in response.json()['response'] if i["role"] == 1]}
                response = session.get(server_api + '/scanner', json={'token': token})
                if response.status_code == 200:
                    scanners = [{"id": scnr["id"], "name": scnr["name"]} for scnr in response.json()['response']]
                else:
                    error_code = response.status_code
            else:
                error_code = response.status_code
            return render_template('users.html', active='users', user=user, users=users, scanners=scanners, error=error_code, message=error_codes.get(response.status_code, ''))
        else:
           return redirect(url_for('main_pages.results_page')) 
    return redirect(url_for('login_page.login'))


@blueprint.route('/home/log/')
def log_page():
    global session
    if request.cookies.get('token'):
        token = request.cookies.get('token')
        user = get_user_info(token)
        if user["role"] == 2:
            response = session.get(server_api + '/scanner', json={'token': token})
            logs = list()
            error_code = None
            message = None
            if response.status_code == 200:
                for scanner in response.json()['response']:
                    for notification in scanner['logs']:
                        logs.append({
                            "name": scanner["name"], 
                            "type": notification["type"], 
                            "message": notification["message"], 
                            "time": datetime.strptime(notification["datetime"], '%d.%m.%Y %H:%M:%S')
                            })
                logs = sorted(logs, key=lambda x: x["time"])
            else:
                error_code = response.status_code
            return render_template('log.html', active='log', logs=logs, user=user, error=error_code, message=error_codes.get(error_code, ''))
        else:
           return redirect(url_for('main_pages.results_page')) 
    return redirect(url_for('login_page.login'))