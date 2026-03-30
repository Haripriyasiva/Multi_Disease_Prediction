import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.types import TypeDecorator, String as SQLAlchemyString

from encryption import Encryption

# Read AES_KEY from env or use default for dev
AES_KEY = os.environ.get('AES_KEY', 'Your32ByteSecretKeyForAES256!!!!').encode('utf-8')
if len(AES_KEY) != 32:
    AES_KEY = b'Your32ByteSecretKeyForAES256!!!!'

encryption = Encryption(AES_KEY)

Base = declarative_base()

class EncryptedString(TypeDecorator):
    impl = SQLAlchemyString
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            import base64
            encrypted_bytes = encryption.encrypt(str(value))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            import base64
            try:
                decrypted = encryption.decrypt(base64.b64decode(value))
                return decrypted
            except Exception:
                return value
        return value

class EncryptedDateTime(TypeDecorator):
    impl = SQLAlchemyString
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            import base64
            if isinstance(value, datetime.datetime):
                str_val = value.isoformat()
            else:
                str_val = str(value)
            encrypted_bytes = encryption.encrypt(str_val)
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            import base64
            try:
                decrypted = encryption.decrypt(base64.b64decode(value))
                return datetime.datetime.fromisoformat(decrypted)
            except Exception:
                try:
                    return datetime.datetime.fromisoformat(str(value))
                except ValueError:
                    return value
        return value


class Patient_Profiles(Base):
    __tablename__ = 'Patient_Profiles'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    family_history = Column(String)
    
    visits = relationship("Patient_Visits", back_populates="patient", cascade="all, delete-orphan")
    audits = relationship("Audit_Log", back_populates="patient", cascade="all, delete-orphan")

class Patient_Visits(Base):
    __tablename__ = 'Patient_Visits'
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey('Patient_Profiles.id'), nullable=False)
    
    visit_date = Column(EncryptedDateTime, default=datetime.datetime.utcnow)
    disease_type = Column(String, nullable=False)
    
    vitals = Column(EncryptedString, nullable=False)
    
    prediction_prob = Column(Float)
    clinical_plan = Column(String)
    chart_image = Column(String)
    
    fasting_glucose = Column(EncryptedString)
    post_prandial_glucose = Column(EncryptedString)
    glucose_delta = Column(EncryptedString)
    hba1c = Column(EncryptedString)
    
    patient = relationship("Patient_Profiles", back_populates="visits")


class Audit_Log(Base):
    __tablename__ = 'Audit_Log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    doctor_id = Column(String, nullable=False)
    action = Column(String, nullable=False)
    patient_id = Column(Integer, ForeignKey('Patient_Profiles.id'))
    
    patient = relationship("Patient_Profiles", back_populates="audits")


DB_PATH = os.environ.get('DB_PATH', 'medical_data.db')
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db_session():
    return SessionLocal()

if __name__ == '__main__':
    init_db()
    print("Database configured with SQLAlchemy and AES-256-GCM encryption.")
