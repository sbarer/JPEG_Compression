from flask import Flask,request
from flask import render_template
from flask_cors import CORS
import json
#Initiate flaskapp 

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_word():
    return 'Hello, World'

@app.route('/compress', methods=['POST'])
def compress():
    req_data = request.get_json()
    compression_value = req_data['compression']
    file_name = req_data['filePath']

    print(compression_value)
    print(file_name)

    data ={}
    data['fileSizeBefore'] = 100
    data['fileSizeAfter'] = 50
    json_data = json.dumps(data)
    return json_data
@app.route('/result')
def result():
    return 'result to be implemented soon!'


if __name__=="__main__":
    app.run(debug=True)