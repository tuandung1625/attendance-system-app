import streamlit as st
from Home import face_rec

st.set_page_config(page_title='Deleting', layout='centered')
st.subheader('Deleting Student or Teacher')

with st.spinner("Retrieving data from Redis database ..."):
    redis_face_db = face_rec.retrieve_data('academy:register')
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved from Redis")

# Nếu redis_face_db là DataFrame
if not redis_face_db.empty:
    members = redis_face_db[['Name', 'Role']].to_dict(orient='records')
    
    # Tạo list hiển thị: "Tên (Role)"
    display_list = [f"{m['Name']} ({m['Role']})" for m in members]
    selected = st.selectbox("Choose member for deleting:", display_list)
    
    if st.button("Delete member"):
        # Find index
        idx = display_list.index(selected)
        member_to_delete = members[idx]
        # Delete
        face_rec.delete_member('academy:register', member_to_delete['Name'], member_to_delete['Role'])
        st.success(f"Delete {member_to_delete['Name']} successfully!")
else:
    st.info("Không có member nào trong database.")