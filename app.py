<<<<<<< HEAD
import streamlit as st
from memory import add_memory, retrieve_memories, summarize_memories, next_turn

st.title("🧠 MemoryFlow AI")

user_input = st.text_input("Enter message:")

if st.button("Submit"):
    t = next_turn()
    add_memory(user_input, t)

    memories = retrieve_memories(user_input)

    st.subheader("Retrieved Memories")
    for m in memories:
        st.write(m)

    st.subheader("Summary")
    st.write(summarize_memories(memories))
=======
import streamlit as st
from memory import add_memory, retrieve_memories, summarize_memories, next_turn

st.title("🧠 MemoryFlow AI")

user_input = st.text_input("Enter message:")

if st.button("Submit"):
    t = next_turn()
    add_memory(user_input, t)

    memories = retrieve_memories(user_input)

    st.subheader("Retrieved Memories")
    for m in memories:
        st.write(m)

    st.subheader("Summary")
    st.write(summarize_memories(memories))
>>>>>>> ddf6092 (Initial MemoryFlow submission)
