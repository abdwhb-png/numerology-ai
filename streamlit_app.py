import os
import uuid
import streamlit as st

from graphs.chat_workflow import graph as chat_bot
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


QUESTIONS_LIMIT = 4


st.set_page_config(page_title=os.getenv("APP_NAME"), page_icon="🤖")


# reset session state
def reset_chat():
    st.session_state.history = []
    st.session_state.question_count = 0
    st.session_state.name = ""
    st.session_state.birth_date = ""
    st.session_state.chat_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.chat_initialized = False


def generate_response(input=""):
    result = chat_bot.invoke(
        {
            "input": input,
            "name": st.session_state.name,
            "birth_date": st.session_state.birth_date,
        },
        config=st.session_state.chat_config,
    )

    return result["generation"]["answer"]


# Initialize session state if it doesn't exist
if "history" not in st.session_state:
    reset_chat()


# sidebar
with st.sidebar:
    st.title(os.getenv("APP_NAME"))

    if os.getenv("APP_ENV") == "local":
        st.success("Test details already provided!", icon="✅")
        st.session_state.name = "John Doe"
        st.session_state.birth_date = "1990-01-01"
    else:
        if not (st.session_state.name and st.session_state.birth_date):
            with st.form(key="user_info_form"):
                st.session_state.name = st.text_input(
                    "Enter Full Name:", type="default"
                )
                st.session_state.birth_date = st.text_input(
                    "Enter Birth Date:", type="default"
                )
                start_chat = st.form_submit_button("Start Chat")

            st.warning(
                "Please provide your details !",
                icon="⚠️",
            )
        else:
            st.success(
                "You can ask up to 3 questions. Proceed by entering your questions!",
                icon="👉",
            )

    st.markdown("📖 Want to know the data used on the model? [knowledge base](#)!")


st.title(os.getenv("APP_NAME"))


if st.session_state.chat_initialized is not True:
    st.write(
        "Please provide your name and birth date to start reading your numerology using this AI chatbot."
    )

    # Initialize chat if name and birth date are provided
    if start_chat and st.session_state.name and st.session_state.birth_date:
        st.session_state.chat_initialized = True
        # Intro
        st.write(f"Welcome {st.session_state.name}, you can start asking questions.")
else:
    # Afficher l'historique des messages
    for message in st.session_state.history:
        if message["role"] == "user":
            st.markdown(f"**🧑 {message['content']}**")
        elif message["role"] == "bot":
            st.markdown(f"**🤖 {message['content']}**")

    # Check questions limit
    while st.session_state.question_count < QUESTIONS_LIMIT:
        if user_input := st.chat_input(
            disabled=not (st.session_state.name and st.session_state.birth_date)
        ):
            # Add user input to history
            st.session_state.history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate a new response if last message is not from assistant
            if st.session_state.history[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_response(user_input)
                        st.write(response)
                message = {"role": "assistant", "content": response}
                # Add chatbot generated response to history
                st.session_state.history.append(message)

                # Increment question count
                st.session_state.question_count += 1

            # Refresh view to show new messages
            st.experimental_rerun()
        else:
            st.write(f"You have reached your limit of {QUESTIONS_LIMIT} questions.")

            # Reset chat button if limit is reached
            if st.button("🔄 Restart chat"):
                reset_chat()
                st.rerun()
