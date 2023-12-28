from flask import Flask, request, render_template, jsonify, send_file
from flask_restful import Api
from resources import AuthApi, UserApi, StatusApi, ScannerApi, ResultApi, VersionApi
import db_session

from utils import get_user, make_error, make_success
from models import User, Result, Scanner


# Application initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sHjla91HdhnBS29QDlshPEUDqG9XzTmDUBO3tSjmHFaKQQwGKl0vD3m1rFD4WV0Y7a7CTJT5wrHIVQF915da6GDHB4qxhbJ9clt4vxlIOu0x0I87bLeB0dyCWQD1iO6S'

# Api initialization
api = Api(app, prefix='/api/v1')

api.add_resource(AuthApi, '/auth')
api.add_resource(UserApi, '/user')
api.add_resource(ScannerApi, '/scanner')
api.add_resource(ResultApi, '/result')
api.add_resource(VersionApi, '/version')
api.add_resource(StatusApi, '/status')


# Get image by barcode and filename
@app.route('/api/v1/image/<int:barcode>/<filename>')
def get_image(barcode, filename):
    session = db_session.create_session()
    res = session.query(Result).filter(Result.barcode == barcode).first()
    session.close()
    return send_file(f"{res.folder_path}/{filename}", mimetype=f"image/{filename.split('.')[-1]}")
    # TODO: check is folder exists


if __name__ == '__main__':
    db_session.global_init()
    app.run(host='0.0.0.0', port=7070)