
# https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3

from flask import Flask, render_template, request, send_file, make_response, url_for, Response, redirect
import pandas as pd
import csv
#import sqlite3
import mysql.connector

#import predict
from predict import run_pred

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("main.html")

@app.route('/custom')
def custom_get():
    return render_template("custom.html")


@app.route('/custom_results', methods=['POST','GET'])
def upload_file_fa():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            #flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            save_p = 'dataset/'+ str(filename)
            file.save(save_p)
            result_p = run_pred(save_p) # run prediction scripts
            rowlist = []
            with open(result_p) as csvfile:
                myreader = csv.reader(csvfile)
                headers = next(myreader, None)
                i = 1
                for row in myreader:
                    rowlist.append(row)
                    i += 1
                    if i >20:
                        return render_template("custom_results.html", resultlist = rowlist, result_path = result_p)
                        break

    #return redirect(url_for('index'))
    return render_template("custom_results.html")


if __name__ == '__main__':
    # run app
    app.run(debug=False, host='0.0.0.0',port=80)
