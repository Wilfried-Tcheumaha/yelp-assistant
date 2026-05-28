import streamlit as st
import requests
from chatbot_ui.core.config import config
from chatbot_ui.utils import YELP_STARS_CSS, render_business_card, render_restaurants_map
import uuid
import logging
import json

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Yelp Assistant", 
    page_icon=":yelp:",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown(YELP_STARS_CSS, unsafe_allow_html=True)

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

session_id = get_session_id()

def api_call(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top right corner of the screen"""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)
        return response.iter_lines(decode_unicode=True)

    except requests.exceptions.ConnectionError:
        _show_error_popup("Could not connect to the API. Please check your connection and try again.")
        return False, {"message": "Connection error."}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return False, {"message": "Request timed out."}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

def api_call_stream(method, url, **kwargs):
    def _show_error_popup(message):
        """Show error message as a popup in the top right corner of the screen"""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.ConnectionError:
        _show_error_popup("Could not connect to the API. Please check your connection and try again.")
        return False, {"message": "Connection error."}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again.")
        return False, {"message": "Request timed out."}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

    return response.iter_lines(decode_unicode=True)

def submit_feedback(feedback_type=None, feedback_text=""):
    """Submit feedback to the API endpoint"""

    def _feedback_score(feedback_type):
        if feedback_type == "positive":
            return 1
        elif feedback_type == "negative":
            return 0
        else:
            return None 
    
    feedback_data = {
        "feedback_score": _feedback_score(feedback_type),
        "feedback_text": feedback_text,
        "trace_id": st.session_state.trace_id,
        "thread_id": session_id,
        "feedback_source_type": "api"
    }

    logger.info(f"Feedback data: {feedback_data}")
    
    url = f"{config.API_URL.rstrip('/')}/feedback/"
    status, response = api_call("post", url, json=feedback_data)
    return status, response


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Yelp business assistant. How can I help you today?"}
    ]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

# Initialize feedback states (simplified)
if "latest_feedback" not in st.session_state:
    st.session_state.latest_feedback = None

if "show_feedback_box" not in st.session_state:
    st.session_state.show_feedback_box = False

if "feedback_submission_status" not in st.session_state:
    st.session_state.feedback_submission_status = None

if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

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
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            is_latest_assistant = (
                message["role"] == "assistant"
                and idx == len(st.session_state.messages) - 1
                and idx > 0
            )

            if not is_latest_assistant:
                continue

            feedback_key = f"feedback_{len(st.session_state.messages)}"
            feedback_result = st.feedback("thumbs", key=feedback_key)

            if feedback_result is not None:
                feedback_type = "positive" if feedback_result == 1 else "negative"

                if st.session_state.latest_feedback != feedback_type:
                    with st.spinner("Submitting feedback..."):
                        status, response = submit_feedback(feedback_type=feedback_type)
                        if status:
                            st.session_state.latest_feedback = feedback_type
                            st.session_state.feedback_submission_status = "success"
                            st.session_state.show_feedback_box = (feedback_type == "negative")
                        else:
                            st.session_state.feedback_submission_status = "error"
                            st.error("Failed to submit feedback. Please try again.")
                    st.rerun()

            if st.session_state.latest_feedback and st.session_state.feedback_submission_status == "success":
                if st.session_state.latest_feedback == "positive":
                    st.success("✅ Thank you for your positive feedback!")
                elif st.session_state.latest_feedback == "negative" and not st.session_state.show_feedback_box:
                    st.success("✅ Thank you for your feedback!")
            elif st.session_state.feedback_submission_status == "error":
                st.error("❌ Failed to submit feedback. Please try again.")

            if st.session_state.show_feedback_box:
                st.markdown("**Want to tell us more? (Optional)**")
                st.caption("Your negative feedback has already been recorded. You can optionally provide additional details below.")

                feedback_text = st.text_area(
                    "Additional feedback (optional)",
                    key=f"feedback_text_{len(st.session_state.messages)}",
                    placeholder="Please describe what was wrong with this response...",
                    height=100,
                )

                col_send, _col_spacer, col_close = st.columns([3, 5, 2])
                with col_send:
                    if st.button("Send Additional Details", key=f"send_additional_{len(st.session_state.messages)}"):
                        if feedback_text.strip():
                            with st.spinner("Submitting additional feedback..."):
                                status, response = submit_feedback(feedback_text=feedback_text)
                                if status:
                                    st.success("✅ Thank you! Your additional feedback has been recorded.")
                                    st.session_state.show_feedback_box = False
                                else:
                                    st.error("❌ Failed to submit additional feedback. Please try again.")
                        else:
                            st.warning("Please enter some feedback text before submitting.")
                        st.rerun()

                with col_close:
                    if st.button("Close", key=f"close_feedback_{len(st.session_state.messages)}"):
                        st.session_state.show_feedback_box = False
                        st.rerun()

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
            status_placeholder = st.empty()
            message_placeholder = st.empty()

            url = f"{config.API_URL.rstrip('/')}/rag/"
            answer: str | None = None

            for line in api_call_stream(
                "post",
                url,
                json={"query": prompt, "thread_id": session_id},
                stream=True,
                headers={"Accept": "text/event-stream"},
            ):
                line_text = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_text.startswith("data:"):
                    continue

                data = line_text[len("data:"):].lstrip()
                if not data:
                    # SSE keep-alive / blank separator — ignore.
                    continue

                # The API emits two kinds of `data:` frames:
                #   1. Plain-string status updates  e.g. "Planning..."
                #   2. A final JSON object         e.g. {"type": "final_answer", ...}
                try:
                    output = json.loads(data)
                except json.JSONDecodeError:
                    output = None

                if isinstance(output, dict) and output.get("type") == "final_answer":
                    payload = output.get("data", {})
                    answer = payload.get("answer", "")
                    st.session_state.trace_id = payload.get("trace_id", "")
                    st.session_state.used_context = payload.get("used_context", [])
                    st.session_state.latest_feedback = None
                    st.session_state.feedback_submission_status = None
                    st.session_state.show_feedback_box = False

                    status_placeholder.empty()
                    message_placeholder.markdown(answer)
                else:
                    # Anything else is a status string ("Analysing the question...",
                    # "Planning...", "Looking for items: ...", etc.).
                    msg = data if output is None else (
                        (output.get("data") or {}).get("message")
                        or output.get("type")
                        or str(output)
                    )
                    status_placeholder.markdown(f"*{msg}*")

            if answer is None:
                answer = "Sorry, the assistant didn't return a response. Please try again."
                status_placeholder.empty()
                message_placeholder.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()