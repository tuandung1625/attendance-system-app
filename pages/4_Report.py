import streamlit as st
from Home import face_rec

st.set_page_config('Reporting', layout='centered')
st.subheader('Reporting')

# Extract data from Redis
name = 'attendance:logs'
def load_logs(name, end=-1):
    logs_list = face_rec.r.lrange(name, start=0,  end=end)
    return logs_list

# tabs to show info
tab1, tab2 = st.tabs(['Registered Data', 'Logs'])
with tab1:
    if st.button('Refresh Data'):
        with st.spinner('Retrieving data from Redis ...'):
            redis_face_db = face_rec.retrieve_data('academy:register')
            st.dataframe(redis_face_db[['Name', 'Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))