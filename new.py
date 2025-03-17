import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize memory to remember past conversations
memory = ConversationBufferMemory()

# Set up AI model (Replace "YOUR_API_KEY" with an actual API key)
ai_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key="AIzaSyAf5dbkHLEn9oIM2tFFftyCeJOcpmOijxg")

def get_ai_response(user_message):
    """Generates a response based on user input."""
    
    # Save user message to memory
    memory.save_context({"user": user_message}, {"ai": "Thinking..."})
    
    # Retrieve previous conversation history
    chat_history = memory.load_memory_variables({})
    
    # Create a prompt for the AI
    prompt = f"""
    You are a data science tutor. Answer ONLY data science-related questions.
    If the question is off-topic, politely decline.
    Maintain context across responses.
    
    Conversation history: {chat_history}
    
    User: {user_message}
    """
    
    # Get response from AI model
    response = ai_model.invoke(prompt)
    
    # Extract the AI's response
    ai_reply = response.get("content", "I'm sorry, I couldn't generate a response.") if isinstance(response, dict) else response.content if hasattr(response, 'content') else str(response)
    
    # Save AI response to memory
    memory.save_context({"user": user_message}, {"ai": ai_reply})
    
    return ai_reply

# Streamlit App


def main():
    st.image(r"C:\Users\shiva\OneDrive\Pictures\photos\AI-and-Data-Science.jpg", width=700)
    st.title("AI Data Science Tutor")
    st.write("Ask me anything about data science!")
    
    # Initialize conversation history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Display previous messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Get user input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        # Store user input in session
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        # Get AI response
        ai_response = get_ai_response(user_input)
        
        # Store AI response in session
        st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)

if __name__ == "__main__":
    main()
