from flask import Flask
import argparse

from login_page import blueprint as login_blueprint
from main_pages import blueprint as main_blueprint
from utils import blueprint as utils_blueprint

app = Flask(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Test', usage='%(test)s [options]')
    parser.add_argument('--debug', type=bool, default=False, help='debug help')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='host help')
    parser.add_argument('--port', type=int, default=5000, help='port help')
    args = vars(parser.parse_args())

    app.register_blueprint(login_blueprint)
    app.register_blueprint(main_blueprint)
    app.register_blueprint(utils_blueprint)
    
    app.run(host=args["host"], port=args["port"])
