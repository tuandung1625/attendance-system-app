import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import os

# connect to redis
hostname = 'redis-10439.c8.us-east-1-2.ec2.redns.redis-cloud.com'
port = 10439
password = 'Sx8lnLad9MgfY6xqrwPEeJsgVGOqtpWT'
r = redis.Redis(
    host=hostname,
    port=port,
    password=password)

# Retrieve data
def retrieve_data(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(), index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['name_role', 'facial_features']
    retrieve_df[['Name', 'Role']] = retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name', 'Role', 'facial_features']]

# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

# ML search algorithm
def ml_search_algorithm(dataframe, embedding, thresh):
    X_list = dataframe['facial_features'].tolist()
    X = np.asarray(X_list)
    y = embedding.reshape(1,-1)

    # calculate cosine similarity
    cosine = pairwise.cosine_similarity(X,y)
    data_search = dataframe.copy()
    data_search['cosine'] = cosine

    # filter the data
    datafilter = data_search.loc[data_search['cosine'] > thresh]
    datafilter.reset_index(drop=True, inplace=True)
    if len(datafilter) > 0:
        argmax = datafilter['cosine'].argmax()
        name, role = datafilter.loc[argmax][['Name', 'Role']]
    else:
        name = 'Unknown'
        role = 'Unknown'
    return name, role


### Real Time Prediction
class RealTimePred():
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])
    
    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])
    
    def savelogs_redis(self):
        dataframe = pd.DataFrame(self.logs)
        dataframe.drop_duplicates('name', inplace=True)

        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)
        
        self.reset_dict()

    def face_prediction(self, test_image, dataframe, thresh):
        # detect time
        current_time = str(datetime.now())

        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embed = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe, embed, thresh)
        
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
            # draw bounding box
            cv2.rectangle(test_copy, (x1,y1), (x2,y2), color, 2)
            text = f'{person_name} : {person_role}'
            cv2.putText(test_copy, text, (x1,y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1,y2+10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


### Registration form 
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    
    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        for res in results:
            self.sample += 1
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            # put text sample info
            text = f"samples={self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX, 0.6,(255,255,0),2)

            # facial features
            embedding = res['embedding']
        
        return frame, embedding

    def save_data_redis(self, name, role):
        # Validation
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # load embedding
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)

        # convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # cal mean
        x_mean = x_array.mean(axis=0)
        x_mean_bytes = x_mean.tobytes()

        # Save into redis
        r.hset(name='academy:register', key=key, value=x_mean_bytes)

        os.remove('face_embedding.txt')
        self.reset()

        return True

# Design deleting function
def delete_member(key, name, role):
    if name is not None:
        if name.strip() != '':
            key_to_del = f'{name}@{role}'
        else:
            return 'name_false'
    else:
        return 'name_false'

    # Delete
    r.hdel(key, key_to_del)

    logs = r.lrange('attendance:logs', 0, -1)
    for log in logs:
        log_str = log.decode()
        if log_str.startswith(f"{name}@{role}@"):
            r.lrem('attendance:logs', 0, log_str)

    return True