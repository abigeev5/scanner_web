import requests

server_api = 'http://127.0.0.1:7070/api/v1'
session = requests.Session()
session.headers.update({'Content-Type': 'application/json'})

error_codes = {
    400: 'Bad request',
    401: 'Unauthorized',
    403: 'No permission',
    404: 'User not found',
    409: 'User already exist',
    500: 'Unknown error'
}