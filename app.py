import streamlit as st
from agent import initialize_messages, get_response

st.set_page_config(
    page_title="Valorant Betting Model Assistant",
    page_icon="🎮"
)

st.title("🎮 Valorant Betting Model Assistant")
st.write("Ask questions about VCT teams, players, matches, events, and analytics.")
st.info("Built for Valorant esports database, workflow, and betting model questions.")

if "messages" not in st.session_state:
    st.session_state.messages = initialize_messages()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🎮").write(msg["content"])

user_input = st.chat_input("Ask your question here...")

if user_input:
    st.chat_message("user", avatar="👤").write(user_input)

    with st.spinner("Thinking..."):
        response, updated_messages = get_response(st.session_state.messages, user_input)
        st.session_state.messages = updated_messages

    st.chat_message("assistant", avatar="🎮").write(response)