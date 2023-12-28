import unittest
import requests
from utils import get_user
import db_session

# curl -H "Content-Type: application/json" -X GET http://192.168.1.126:7070/api/v1/auth -H "Authorization:Bearer test"
# curl -H "Content-Type: application/json" -X GET http://192.168.1.126:7070/api/v1/auth -d "{\"username\": \"abigeev\", \"password\": \"test\"}"
# curl -H "Content-Type: application/json" -X GET http://192.168.1.126:7070/api/v1/user -d "{\"token\": \"a27d94e52cfd2af22a0b4232648ad92f7a4a29ee683e6013b348c133605adb1e\"}"

api = 'http://192.168.1.126:7070/api/v1/'
headers = {'Content-Type': 'application/json'}

adminLogin = 'testAdmin'
adminPassword = 'admin'
adminToken = '1e44082e74b89e2d06975aa7dcaaa399b6f6098586e0de5636c424b578400f31'

userLogin = 'testUser'
userPassword = 'user'
userToken = '7a3d2c03038e1a7fcc33a7a8f0515b86879bbedc39c102ce5e87063b8dcd2336'


class TestAuthApi(unittest.TestCase):

	# Normal login
	def test_get_token_1(self):
		response = requests.get(api + 'auth', json={'username': userLogin, 'password': userPassword}, headers=headers)
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()['token'], userToken)

	# Wrong password
	def test_get_token_2(self):
		response = requests.get(api + 'auth', json={'username': userLogin, 'password': 'ASJndmfnkj'}, headers=headers)
		self.assertEqual(response.status_code, 401)

	# Unexisting account
	def test_get_token_3(self):
		response = requests.get(api + 'auth', json={'username': 'Ajklfdsji', 'password': 'ASJndmfnkj'}, headers=headers)
		self.assertEqual(response.status_code, 404)

	# Bad request
	def test_get_token_4(self):
		response = requests.get(api + 'auth', json={'username': userLogin}, headers=headers)
		self.assertEqual(response.status_code, 400)


class TestUserApi(unittest.TestCase):
	# Create user
	@unittest.skip("SQL Latecy problem")
	def test_create_user_1(self):
		db_session.global_init()
		response = requests.post(api + 'user', json={
			'token': adminToken,
			'username': 'TestAccontForTest1427',
			'password': 'test'}, headers=headers)
		self.assertEqual(response.status_code, 200)
		self.assertNotEqual(get_user(username='TestAccontForTest1427'), None)

	# Existing account
	def test_create_user_2(self):
		db_session.global_init()
		response = requests.post(api + 'user', json={
			'token': adminToken,
			'username': userLogin,
			'password': 'test'}, headers=headers)
		self.assertEqual(response.status_code, 409)
		self.assertNotEqual(get_user(username=userLogin), None)

	# No permission
	def test_create_user_3(self):
		db_session.global_init()
		response = requests.post(api + 'user', json={
			'token': userToken,
			'username': 'TestAccontForTest1427',
			'password': 'test'}, headers=headers)
		self.assertEqual(response.status_code, 403)
		self.assertNotEqual(get_user(token=userToken), 2)

	# Delete user
	@unittest.skip("SQL Latecy problem")
	def test_delete_user_3(self):
		db_session.global_init()
		response = requests.delete(api + 'user', json={
			'token': adminToken,
			'username': 'TestAccontForTest1427'}, headers=headers)
		self.assertEqual(response.status_code, 200)
		self.assertEqual(get_user(username='TestAccontForTest1427'), None)


class TestScannerApi(unittest.TestCase):
	def test_get_image(self):
		pass


# py test.py -v
# Executing the tests in the above test case class
if __name__ == "__main__":
	db_session.global_init()
	unittest.main()