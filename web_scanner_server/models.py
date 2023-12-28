from db_session import SqlAlchemyBase
from sqlalchemy import Column, String, Integer, DATETIME, ForeignKey, JSON, BOOLEAN
from sqlalchemy.orm import relationship


class User(SqlAlchemyBase):
	__tablename__ = 'users'
	
	id = Column(Integer, primary_key=True)
	username = Column(String)
	info = Column(String)
	name = Column(String)
	password = Column(String)
	token = Column(String)
	role = Column(Integer)

	def from_json(self, data):
		if "username" in data:
			self.username = data.get("username", "")
		if "info" in data:
			self.info = data.get("info", {})
		if "name" in data:
			self.name = data.get("name", "")
		if "password" in data:
			self.password = data.get("password", "")
		if "token" in data:
			self.token = data.get("token", "")
		if "role" in data:
			self.role = data.get("role", 1)
   
	def to_json(self):
		return {
      		"id": self.id,
			"username": self.username,
			"info": self.info,
			"name": self.name,
			"password": self.password,
			"token": self.token,
			"role": self.role
		}

class Scanner(SqlAlchemyBase):
	__tablename__ = 'scanners'
	
	id = Column(Integer, primary_key=True)
	name = Column(String)
	version = Column(Integer)
	info = Column(JSON)
	settings = Column(JSON)
	operators = Column(JSON)
	logs = Column(JSON)

	def from_json(self, data):
		if "name" in data:
			self.name = data.get("username", "")
		if "version" in data:
			self.version = data.get("version", "")
		if "info" in data:
			self.info = data.get("info", {})
		if "settings" in data:
			self.settings = data.get("settings", {})
		if "operators" in data:
			self.operators = data.get("operators", "")
		if "logs" in data:
			self.logs = data.get("logs", {})
   
   
	def to_json(self):
		return {
      		"id": self.id,
			"name": self.name,
			"version": self.version,
			"info": self.info,
			"settings": self.settings,
			"operators": self.operators,
			"logs": self.logs
		}


class Result(SqlAlchemyBase):
	__tablename__ = 'results'
	
	id = Column(Integer, primary_key=True)
	user_id = Column(Integer, ForeignKey('users.id'))
	scanner_id = Column(Integer, ForeignKey('scanners.id'))
	barcode = Column(Integer)
	date = Column(DATETIME)
	results = Column(JSON)
	enterobiasis = Column(BOOLEAN)
	folder_path = Column(String)

	def from_json(self, data):
		if "user_id" in data:
			self.user_id = data["user_id"]
		if "scanner_id" in data:
			self.scanner_id = data["scanner_id"]
		if "barcode" in data:
			self.barcode = data["barcode"]
		if "date" in data:
			self.date = data["date"]
		if "results" in data:
			self.results = data["results"]
		if "enterobiasis" in data:
			self.enterobiasis = data["enterobiasis"]
		if "folder_path" in data:
			self.folder_path = data["folder_path"]


	def to_json(self):
		return {
      		"id": self.id,
			"user_id": self.user_id,
			"scanner_id": self.scanner_id,
			"barcode": self.barcode,
			"date": self.date,
			"results": self.results,
			"enterobiasis": self.enterobiasis,
			"folder_path": self.folder_path
		}
   

class Version(SqlAlchemyBase):
	__tablename__ = 'versions'
    
	id = Column(Integer, primary_key=True)
	name = Column(String)
	info = Column(JSON)
	settings = Column(JSON)
 
	def from_json(self, data):
		if "settings" in data:
			self.settings = data["settings"]
		if "info" in data:
			self.info = data["info"]
		if "name" in data:
			self.name = data["name"]
   
	def to_json(self):
		return {
			"id": self.id,
			"settings": self.settings,
			"info": self.info,
			"name": self.name
	   }