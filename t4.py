import streamlit as st
import json
import time
import requests  # Add this import for making HTTP requests to Ollama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get configuration from .env file
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://192.168.124.100:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3-cot')


def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2
                    }
                }
            )
            response.raise_for_status()
            print(response.json())
            return json.loads(response.json()["message"]["content"])
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(2)  # Wait for 1 second before retrying


def generate_response(prompt):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data['next_action'] == 'final_answer':
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time


def main():
    st.set_page_config(page_title="ol1 prototype - Ollama version", page_icon="🧠", layout="wide")

    st.title("ol1: Using Ollama to create o1-like reasoning chains")

    st.markdown("""
    This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated. It is powered by Ollama so that the reasoning step is local!

    Forked from [bklieger-groq](https://github.com/bklieger-groq)
    Open source [repository here](https://github.com/win4r/o1)
    """)

    st.markdown(f"**Current Configuration:**")
    st.markdown(f"- Ollama URL: `{OLLAMA_URL}`")
    st.markdown(f"- Ollama Model: `{OLLAMA_MODEL}`")

    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        st.write("Generating response...")

        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)

            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")


if __name__ == "__main__":
    main()