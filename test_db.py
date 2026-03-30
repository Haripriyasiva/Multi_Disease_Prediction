from database import init_db, SessionLocal, Patient_Profiles, Patient_Visits
import datetime

init_db()
db = SessionLocal()

p = Patient_Profiles(username="test_user", name="Test User", family_history="")
db.add(p)
db.commit()

v = Patient_Visits(
    patient_id=p.id,
    disease_type="Diabetes",
    vitals='{"test": 123}',
    fasting_glucose="100",
    hba1c="5.5",
    visit_date=datetime.datetime.utcnow()
)
db.add(v)
db.commit()

v_fetched = db.query(Patient_Visits).filter_by(patient_id=p.id).first()
print("Vitals:", v_fetched.vitals, type(v_fetched.vitals))
print("Date:", v_fetched.visit_date, type(v_fetched.visit_date))
print("Glucose:", v_fetched.fasting_glucose)
