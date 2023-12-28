import sqlalchemy
import sqlalchemy.orm as orm
from sqlalchemy.orm import Session
import sqlalchemy.ext.declarative as dec

SqlAlchemyBase = dec.declarative_base()

__factory = None
engine = None

def global_init():
    global __factory
    global engine

    if __factory:
        return

    conn_str = 'mysql+pymysql://admin:Test@localhost/scanner?charset=utf8mb4'

    engine = sqlalchemy.create_engine(conn_str, pool_size=20, max_overflow=0, echo=False)
    __factory = orm.sessionmaker(bind=engine)

    SqlAlchemyBase.metadata.create_all(engine)


def get_engine():
    global engine
    return engine


def create_session() -> Session:
    global __factory
    return __factory()