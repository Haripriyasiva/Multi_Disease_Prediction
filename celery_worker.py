import os
from celery import Celery
import time
from database import get_db_session, Patient_Visits, Notification

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app = Celery('tasks', broker=REDIS_URL)

@app.task
def schedule_post_prandial_notification(visit_id):
    # Reduced from 7200 seconds (2 hours) to 10 seconds so you can see it in the frontend!
    time.sleep(10)

    db = get_db_session()
    visit = db.query(Patient_Visits).filter(Patient_Visits.id == visit_id).first()

    if visit and not visit.post_prandial_glucose:
        patient_id = visit.patient_id
        app.send_task('celery_worker.send_push_notification', args=[patient_id, "Time for your Post-Meal reading. Log it now to complete your daily profile."])
    
    db.close()

@app.task
def send_push_notification(patient_id, message):
    print(f"Sending push notification to patient {patient_id}: {message}")
    db = get_db_session()
    try:
        new_notification = Notification(patient_id=patient_id, message=message)
        db.add(new_notification)
        db.commit()
    except Exception as e:
        print(f"Error saving notification: {e}")
    finally:
        db.close()
