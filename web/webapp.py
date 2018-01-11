from flask import Flask, render_template, request

from task1 import getResults
from task2 import *
from task2 import getLocationList

import config
app = Flask(__name__)

locations = getLocationList()


@app.route('/')
def student():
    return render_template('search.html', locs=locations)


@app.route('/search', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form

        images = getResults(result["location"])
        images = ["http://" + ONLINE_TEST_PATH + "img/" + result["location"] + "/" + str(i) + ".jpg" for i in images]
        if int(result["number"]) > 0:
            images = images[:int(result["number"])]

        imgs = list(range(len(images)))
        for i in range(len(images)):
            imgs[i] = [images[i]]

        return render_template("search.html", data=result, locs=locations, imgs=imgs)


if __name__ == '__main__':
    app.run(threaded=True)
