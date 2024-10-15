from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    summary = summarizer(data["text"], max_length=150, min_length=40, do_sample=False)
    return jsonify(summary[0]["summary_text"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
