from flask import Flask, render_template, request

from utility_functions import *

app = Flask(__name__)

locations = getLocationList()


@app.route('/')
def student():
    return render_template('search.html', locs=locations)


@app.route('/search', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form

        images = getImages(result["location"])
        images = reorderImages(dict(zip(images, range(len(images)))), result["location"])
        cluster_data = getClusterData(result["location"])

        clusters = []
        for i in images:
            clusters.append("c" + cluster_data[int(i)])

        images = ["http://" + ONLINE_TEST_PATH + "img/" + result["location"] + "/" + i + ".jpg" for i in images]
        if int(result["number"]) > 0:
            images = images[:int(result["number"])]

        imgs = list(range(len(images)))
        for i in range(len(images)):
            imgs[i] = [images[i], clusters[i]]

        return render_template("search.html", data=result, locs=locations, imgs=imgs)


if __name__ == '__main__':
    app.run(threaded=True)
