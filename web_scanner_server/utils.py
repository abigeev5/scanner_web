# from flask_httpauth import HTTPBasicAuth, HTTPDigestAuth, HTTPTokenAuth
# from werkzeug.exceptions import BadRequest
from flask import request, jsonify, make_response
from models import User
import db_session
import functools
import secrets

error_codes = {
    400: 'Bad request',
    401: 'Unauthorized',
    403: 'No permission',
    404: 'User not found',
    409: 'User already exist',
    500: 'Unknown error'
}


def get_user(id=None, username=None, token=None):
    session = db_session.create_session()
    response = None
    if not(id is None):
        response = session.query(User).filter(User.id == id).first()
    elif not(token is None):
        response = session.query(User).filter(User.token == token).first()
    else:
        response = session.query(User).filter(User.username == username).first()
    session.close()
    return response


def make_error(code):
    return make_response(jsonify({'Error': error_codes[code]}), code)

def make_success(code):
    return make_response(jsonify({'Result': 'Ok'}), code)

def get_error_json(code):
    return ({'Error': error_codes[code]}, code)

def create_token(length=64):
    return secrets.token_hex(length)


# decorator to parse json requests and handle errors
def verify_token(f=None, decrypt=True, encrypt=True, role=None, required_args=[]):
    assert callable(f) or f is None
    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            try:
                data = request.get_json()
            except Exception as e:
                print('[ERROR]', e)
                return make_error(500)
            if not all([arg in data for arg in required_args]):
                return make_error(400)
            user = None
            if ('token' in required_args):
                user = get_user(token=data['token'])
                if user:
                    if not(role is None):
                        if user.role < role:
                            return make_error(403)
                else:
                    return make_error(401)
            return func(*args, user, data)
        return wrapper
    return _decorator(f) if callable(f) else _decorator