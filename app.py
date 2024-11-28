from flask import Flask, render_template, request, jsonify
from new import load_faq_data, get_answer_for_query

app = Flask(__name__)

# Load and preprocess the FAQ data
pdf_path = "./FAQ.pdf"  # Adjust the path to your PDF
vectorizer, question_vectors, questions, qa_dict = load_faq_data(pdf_path)

# Initialize conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html', conversation=conversation_history)

@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.json.get("message")
    
    if user_query:
        # Append the user's message to the conversation history
        conversation_history.append({"sender": "user", "message": user_query})
        
        # Get the assistant's answer
        answer = get_answer_for_query(user_query, vectorizer, question_vectors, questions, qa_dict)
        
        # Append the assistant's response to the conversation history
        conversation_history.append({"sender": "assistant", "message": answer})
        
        return jsonify({"status": "success", "message": answer})
    return jsonify({"status": "error", "message": "No input received"})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    assistant_message = data.get("message")
    feedback_type = data.get("feedback")  # 'like' or 'dislike'

    # Save feedback to a log file or database
    with open("feedback_log.txt", "a") as feedback_file:
        feedback_file.write(f"Message: {assistant_message}\nFeedback: {feedback_type}\n\n")

    return jsonify({"status": "success", "message": "Feedback received!"})

if __name__ == '__main__':
    app.run(debug=True)
