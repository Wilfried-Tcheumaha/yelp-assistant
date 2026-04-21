import streamlit as st
import requests
from chatbot_ui.core.config import config
from chatbot_ui.utils import YELP_STARS_CSS, render_business_card, render_restaurants_map

st.set_page_config(
    page_title="Yelp Assistant", 
    page_icon=":yelp:",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown(YELP_STARS_CSS, unsafe_allow_html=True)



def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top right corner of the screen"""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid JSON response from API"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Could not connect to the API. Please check your connection and try again.")
        return False, {"message": "Connection error."}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return False, {"message": "Request timed out."}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Yelp business assistant. How can I help you today?"}
    ]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

with st.sidebar:
    suggestions_tab, = st.tabs(["Suggestions"])
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.markdown(render_business_card(item), unsafe_allow_html=True)
                st.divider()
        else:
            st.info("No suggestions yet")

chat_col, map_col = st.columns([2, 1], gap="large")

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with map_col:
    if st.session_state.used_context:
        render_restaurants_map(st.session_state.used_context)

# `st.chat_input` must stay at page-level (it pins to the bottom of the page),
# so we render the new messages inside `chat_col` ourselves before the rerun.
if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            url = f"{config.API_URL.rstrip('/')}/rag/"
            ok, data = api_call("post", url, json={"query": prompt})

            st.session_state.used_context = data.get("used_context", [])
            # Success: API returns {"answer", "used_context"}. Errors: {"detail": ...}.
            if ok:
                answer = data.get("answer", "")
            else:
                err = data.get("detail", data.get("message", "Request failed"))
                answer = err if isinstance(err, str) else str(err)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()