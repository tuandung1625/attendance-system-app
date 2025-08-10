import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Predictions', layout='centered')
st.subheader('Real-Time Attendance System')

# Retrieve the data from Database
with st.spinner("Retrieving data from Redis database ..."):
    redis_face_db = face_rec.retrieve_data('academy:register')
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved from Redis")

# time
waitTime = 30
setTime = time.time()
realtimepred = face_rec.RealTimePred()

# Real Time Prediction
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format='bgr24')  # 3 dimension numpy array
    pred_frame = realtimepred.face_prediction(img, redis_face_db, 0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.savelogs_redis()
        setTime = time.time()
        print("Save data to redis")
        
    return av.VideoFrame.from_ndarray(pred_frame, format='bgr24')

webrtc_streamer(key='realtimePrediction', video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})