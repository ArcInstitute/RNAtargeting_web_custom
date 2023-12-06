
# import
import os
import csv
from werkzeug.exceptions import HTTPException
from flask import Flask, render_template, request, url_for, redirect
#import pandas as pd
#import sqlite3
#import mysql.connector

#import predict
from predict import run_pred

# Init app
app = Flask(__name__)

# Set home page
@app.route('/')
def homepage():
    return render_template("main.html")

# Set custom primer page
@app.route('/custom')
def custom_get():
    return render_template("custom.html")

# Set custom primer results page
@app.route('/custom_results', methods=['POST','GET'])
def upload_file_fa():
    if request.method == "POST":
        # check if the post request has the file object
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(url_for('custom_get'))
        # save the file
        save_p = os.path.join('dataset', str(file.filename))
        file.save(save_p)
        # run prediction
        result_p = run_pred(save_p) # run prediction scripts
        # render results
        rowlist = []
        with open(result_p) as csvfile:
            myreader = csv.reader(csvfile)
            headers = next(myreader, None)
            for i,row in enumerate(myreader):
                rowlist.append(row)
                if i > 20:
                    return render_template(
                        "custom_results.html", 
                        resultlist = rowlist, 
                        result_path = result_p
                    )
    return render_template("custom_results.html")


@app.errorhandler(HTTPException)
def page_not_found(e):
    return render_template('http_error.html')

if __name__ == '__main__':
    # run app
    app.run(debug=True, host='0.0.0.0', port=8080)
