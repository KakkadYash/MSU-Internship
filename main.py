import os
import json 
import logging
import traceback
import bcrypt
import cv2
import numpy as np
import logging
import mediapipe as mp
import tensorflow_hub as hub
from datetime import timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from itsdangerous import URLSafeTimedSerializer, BadData
from google.oauth2 import service_account
from google.cloud import storage, secretmanager, firestore
from tensorflow import keras
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY"))
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
bucket_name = 'video_data_bucket_001'
_cached_credentials = None

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# Firestore client
db = firestore.Client()

# GCS signed URL support
def get_signed_url_credentials():
    global _cached_credentials
    if _cached_credentials:
        return _cached_credentials
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project_id}/secrets/SIGNED_URL_KEY/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    info = json.loads(response.payload.data.decode("utf-8"))
    _cached_credentials = service_account.Credentials.from_service_account_info(info)
    return _cached_credentials

def generate_signed_url(blob_name):
    credentials = get_signed_url_credentials()
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=15),
        method="GET"
    )

logging.basicConfig(level=logging.INFO)

# Load models
try:
    logging.info("Loading height estimation model...")
    model = keras.models.load_model("height_estimation_model.h5")
    logging.info("Height estimation model loaded.")
except Exception as e:
    logging.exception("Failed to load height model")

try:
    logging.info("Loading MoveNet model from TFHub...")
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    movenet = movenet.signatures['serving_default']
    logging.info("MoveNet model loaded.")
except Exception as e:
    logging.exception("Failed to load MoveNet model")

def extract_landmarks(image):
    with mp.solutions.pose.Pose() as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(99)

def calculate_additional_features(landmarks):
    return np.array([np.linalg.norm(landmarks[0:3] - landmarks[3:6])])

@app.post("/estimate_height")
async def estimate_height(video: UploadFile = File(...)):
    try:
        video_path = f"/tmp/{video.filename}"
        with open(video_path, 'wb') as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(video_path)
        frames, count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 5 == 0:
                lms = extract_landmarks(frame)
                features = np.concatenate((lms, calculate_additional_features(lms)))
                frames.append(features)
            count += 1
        cap.release()
        os.remove(video_path)

        if not frames:
            raise HTTPException(400, "No valid landmarks detected")
        preds = model.predict(np.mean(np.array(frames), axis=0).reshape(1, -1))
        return {"estimated_height": float(np.mean(preds))}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to estimate height: {str(e)}")

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    name: str
    username: str
    email: str
    password: str

class ResetRequest(BaseModel):
    email: str

class NewPasswordRequest(BaseModel):
    password: str

@app.post("/signup")
def signup(payload: SignupRequest):
    hashed = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()
    doc_ref = db.collection("users").add({
        "name": payload.name,
        "username": payload.username,
        "email": payload.email,
        "password": hashed
    })
    return {"message": "User registered successfully", "userId": doc_ref[1].id}

@app.post("/login")
def login(payload: LoginRequest):
    users_ref = db.collection("users")
    docs = users_ref.where("username", "==", payload.username).get()
    if not docs:
        docs = users_ref.where("email", "==", payload.username).get()
    if not docs:
        raise HTTPException(404, "User not found")
    user_data = docs[0].to_dict()
    if not bcrypt.checkpw(payload.password.encode(), user_data['password'].encode()):
        raise HTTPException(401, "Invalid password")
    return {"message": "Login successful", "userId": docs[0].id, "username": user_data['username']}

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    thumbnail: UploadFile = File(None),
    userId: str = Form(...),
    uploadDate: str = Form(None)
):
    try:
        local_path = f"/tmp/{video.filename}"
        with open(local_path, 'wb') as f:
            f.write(await video.read())

        gcs_video_name = f'videos/{userId}/{video.filename}'
        storage_client = storage.Client()
        video_blob = storage_client.bucket(bucket_name).blob(gcs_video_name)
        video_blob.chunk_size = 5 * 1024 * 1024
        video_blob.upload_from_filename(local_path)
        os.remove(local_path)

        gcs_image_name = None
        if thumbnail:
            thumb_path = f"/tmp/{thumbnail.filename}"
            with open(thumb_path, 'wb') as f:
                f.write(await thumbnail.read())
            gcs_image_name = f'images/{userId}/{thumbnail.filename}'
            storage_client.bucket(bucket_name).blob(gcs_image_name).upload_from_filename(thumb_path)
            os.remove(thumb_path)

        doc_ref = db.collection("videos").add({
            "userId": userId,
            "videoName": video.filename,
            "filePath": gcs_video_name,
            "thumbnailPath": gcs_image_name,
            "uploadDate": uploadDate
        })
        return {"message": "Video uploaded successfully", "videoId": doc_ref[1].id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error uploading video: {str(e)}")

# ----- Add /forgot_password -----
@app.post("/forgot_password")
def forgot_password(payload: ResetRequest):
    """
    Sends a password-reset email if the email exists.
    (Response is intentionally the same whether or not the email exists.)
    """
    try:
        # Find user doc by email (if any)
        users = db.collection("users").where("email", "==", payload.email).get()
        if not users:
            # Keep behavior same as before: don't leak existence of email
            return {"message": "If this email exists, a reset link will be sent"}

        # Create token and reset URL
        token = serializer.dumps(payload.email, salt='password-reset-salt')
        reset_url = f"https://kakkadyash.github.io/src/pages/reset_password.html?token={token}"

        # Build email
        message = MIMEMultipart("alternative")
        message["Subject"] = "Password Reset Request"
        message["From"] = EMAIL_USER
        message["To"] = payload.email

        text = f"Click to reset your password: {reset_url}"
        html = f"<html><body><p>Reset your password by clicking <a href='{reset_url}'>here</a>.</p></body></html>"

        message.attach(MIMEText(text, "plain"))
        message.attach(MIMEText(html, "html"))

        # Send email (Gmail SMTP)
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, payload.email, message.as_string())

        return {"message": "If this email exists, a reset link will be sent"}
    except Exception as e:
        logging.exception("Failed in forgot_password")
        # Don't expose internal error details to client
        raise HTTPException(500, "Failed to send reset email")


# ----- Add /reset_password -----
@app.post("/reset_password/{token}")
def reset_password(token: str = Path(...), payload: NewPasswordRequest = None):
    """
    Accepts a token in the URL and new password in the JSON body:
    { "password": "newpass123" }
    """
    try:
        # Verify token and extract email (token expires in 3600s)
        try:
            email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
        except BadData:
            raise HTTPException(400, "Invalid or expired token")

        if payload is None or not getattr(payload, "password", None):
            raise HTTPException(400, "Password is required")

        # Find user by email
        users = db.collection("users").where("email", "==", email).get()
        if not users:
            raise HTTPException(404, "User not found")

        hashed_password = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()

        # Update the first matching user doc's password
        user_doc = users[0]
        db.collection("users").document(user_doc.id).update({"password": hashed_password})

        return {"message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Failed in reset_password")
        raise HTTPException(500, "Failed to reset password")
    
@app.post("/saveAnalytics")
def save_analytics(data: dict):
    required_fields = [
        'video_Id', 'userId', 'idealHeadPercentage', 'topSpeed',
        'averageAthleticScore', 'averageJumpHeight',
        'averageStrideLength', 'peakAcceleration', 'peakDeceleration'
    ]
    if any(data.get(field) is None for field in required_fields):
        raise HTTPException(400, "Missing required analytics fields")

    db.collection("analytics").add({
        "videoId": data['video_Id'],
        "userId": data['userId'],
        "idealHeadPercentage": data['idealHeadPercentage'],
        "topSpeed": data['topSpeed'],
        "averageAthleticScore": data['averageAthleticScore'],
        "averageJumpHeight": data['averageJumpHeight'],
        "averageStrideLength": data['averageStrideLength'],
        "peakAcceleration": data['peakAcceleration'],
        "peakDeceleration": data['peakDeceleration']
    })
    return {"message": "Analytics saved successfully"}

@app.get("/history")
def get_history(userId: str):
    if not userId:
        raise HTTPException(400, "User ID is required")
    try:
        videos = db.collection("videos").where("userId", "==", userId).order_by("uploadDate", direction=firestore.Query.DESCENDING).get()
        history = []
        for vid in videos:
            vdata = vid.to_dict()
            analytics_docs = db.collection("analytics").where("videoId", "==", vid.id).get()
            analytics_data = analytics_docs[0].to_dict() if analytics_docs else {}
            vdata.update(analytics_data)
            vdata['video_url'] = generate_signed_url(vdata['filePath']) if vdata.get('filePath') else None
            vdata['thumbnail_url'] = generate_signed_url(vdata['thumbnailPath']) if vdata.get('thumbnailPath') else None
            history.append(vdata)
        return {"history": history}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to fetch history: {str(e)}")

@app.get("/get-total-uploads")
def get_total_uploads(userId: str):
    videos = db.collection("videos").where("userId", "==", userId).get()
    return {"total_uploads": len(videos)}

@app.get("/profile")
def profile(userId: str):
    doc = db.collection("users").document(userId).get()
    if not doc.exists:
        raise HTTPException(404, "User not found")
    return doc.to_dict()

@app.post("/updateProfile")
def update_profile(data: dict):
    required_fields = ['userId', 'age', 'state', 'sports']
    if any(data.get(field) is None for field in required_fields):
        raise HTTPException(400, "All fields are required")
    if not isinstance(data['sports'], list):
        raise HTTPException(400, "Sports should be an array of strings")

    sports_str = ', '.join(data['sports'])
    db.collection("users").document(data['userId']).update({
        "age": data['age'],
        "state": data['state'],
        "sports": sports_str
    })
    return {"message": "Profile updated successfully"}

@app.get("/")
def root():
    return {"message": "FastAPI app is up"}






