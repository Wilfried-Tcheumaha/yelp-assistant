import streamlit as st
import requests
from chatbot_ui.core.config import config



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
    st.session_state.messages=[{"role": "assistant", "content": "Hello! I'm your Yelp business assistant. How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        url = f"{config.API_URL.rstrip('/')}/rag/"
        ok, data = api_call("post", url, json={"query": prompt})
        # Success: API returns {"answer", "request_id"}. Errors: {"detail": ...} — no "answer" key.
        if ok:
            answer = data.get("answer", "")
        else:
            err = data.get("detail", data.get("message", "Request failed"))
            answer = err if isinstance(err, str) else str(err)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})