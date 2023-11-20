from flask import Flask, render_template, request, jsonify
import google.generativeai as palm

app = Flask(__name__)

API_KEY = 'AIzaSyA4Q2RuZuU5p65TOupateJOatfm_jYYXqA'
palm.configure(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')
a = 0
@app.route('/ask', methods=['POST'])
def ask():
    global a
    user_message = request.form['user_message']
    response = palm.chat(messages=user_message, temperature=0.1, context=' Speak like a certified trainer and dietician,take inputs such as ask him about age,diet, and other relevent features to sugesst him diet as per he needs')
    chat_response = response.last

    if(a==0):
        chat_response = "I can suggest you iteneraries and places to visit ??"
        a=1
    return jsonify({'response': chat_response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)