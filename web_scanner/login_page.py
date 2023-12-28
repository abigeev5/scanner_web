from flask import render_template, request, redirect, url_for, make_response, Blueprint
import hashlib 

from config import server_api, session, error_codes
from utils import get_user_info


blueprint = Blueprint('login_page', __name__, template_folder='templates')


@blueprint.route('/login')
def login():
    return render_template('login.html')


@blueprint.route('/login', methods=['POST'])
def login_post():
    # userLogin = 'testUser'
    # userPassword = 'user'
    username = request.form.get('username')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    password = hashlib.sha256(password.encode()).hexdigest()
    
    response = session.get(server_api + '/auth', json={'username': username, 'password': password})
    if response.status_code == 200:
        token = response.json()['token']
        res = make_response(redirect(url_for('main_pages.home', scanner_id=0, results_id=None, token=token)))
        age = 60*24*31 if remember else 60*24
        res.set_cookie('token', token, max_age=age)
        return res
    return render_template('login.html', error=response.status_code, message=error_codes.get(response.status_code, ''))


@blueprint.route('/logout')
def logout():
    res = make_response(redirect(url_for('login_page.login')))
    token = request.cookies.get('token')
    res.set_cookie('token', token, max_age=0)
    return res