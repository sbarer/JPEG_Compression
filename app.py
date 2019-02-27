from flask import Flask,request
from flask import render_template
from flask_cors import CORS
import json
from jpeg_coder import *

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
    compression_value = int(compression_value) / int(10)
    print(compression_value)
    jCoder = jpeg_coder(0.5)
    file_name = './src/assets/images/' + str(file_name)
    sizeBefore, sizeAfter = jCoder.encode(file_name)


    print(compression_value)
    print(file_name)
    print(int(sizeBefore))
    print(int(sizeAfter))

    data ={}
    data['fileSizeBefore'] = int(sizeBefore)
    data['fileSizeAfter'] = int(sizeAfter)
    json_data = json.dumps(data)
    return json_data
    
@app.route('/result')
def result():
    return 'result to be implemented soon!'


if __name__=="__main__":
    app.run(debug=True)