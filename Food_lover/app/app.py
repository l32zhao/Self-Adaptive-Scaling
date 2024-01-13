import requests
import hashlib
import uuid
import logging
import io
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    jsonify,
    url_for,
    send_file,
)
from random import randrange
from flask import Flask, render_template, flash, request, redirect, url_for
from datetime import datetime
import os
from flask import Flask, request, render_template, redirect, url_for
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from bson import ObjectId
from gridfs import GridFS
from flask_mysqldb import MySQL
from pymongo import MongoClient


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app.config["MYSQL_HOST"] = "db"
app.config["MYSQL_USER"] = "root"
app.config["PORT"] = "3306"
app.config["MYSQL_PASSWORD"] = "root"
app.config["MYSQL_DB"] = "fdlover"
mysql = MySQL(app)

client = MongoClient(
    host="mongo",
    port=27017,
    username="guanjie",
    password="guanjie",
    authSource="admin",
)
mongo = client["fdlover"]

gridfs_bf = GridFS(mongo, collection="breakfast")
gridfs_lu = GridFS(mongo, collection="lunch")
gridfs_dn = GridFS(mongo, collection="dinner")
gridfs_sn = GridFS(mongo, collection="snack")


@app.route("/", methods=["GET", "POST"])
@app.route("/welcome", methods=["GET", "POST"])
def welcome():
    return render_template("main.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        password = password + "salt"
        encrypted_password = str(hashlib.md5(password.encode()).hexdigest())
        cursor = mysql.connection.cursor()
        table_name = "users"
        query = f"SELECT * FROM {table_name} WHERE username = %s and password = %s"
        cursor.execute(query, (username, encrypted_password))
        accout = cursor.fetchone()
        if accout:
            temp = url_for("upload") + "?username=" + str(username)
            cursor.close()
            return render_template("register.html", message="successful", new_page=temp)
        else:
            temp = url_for("login")
            cursor.close()
            return render_template("register.html", error="error", new_page=temp)

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        password = password + "salt"
        encrypted_password = str(hashlib.md5(password.encode()).hexdigest())
        cursor = mysql.connection.cursor()

        table_name = "users"
        query = f"SELECT * FROM {table_name} WHERE username = %s"
        cursor.execute(query, (username,))
        account = cursor.fetchone()

        if account:
            temp = url_for("login") + "?username=" + str(username)
            cursor.close()
            return render_template(
                "login.html", message="Account already exists", new_page=temp
            )
        else:
            table_name = "users"
            query = f"INSERT INTO {table_name} (username, password) VALUES (%s, %s)"
            cursor.execute(query, (username, encrypted_password))
            mysql.connection.commit()  # Commit the changes to the database
            temp = url_for("upload") + "?username=" + str(username)
            cursor.close()
            return render_template(
                "register.html", message="Registration successful", new_page=temp
            )

    return render_template("register.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    username = request.args.get("username", "")
    if request.method == "POST":
        recipFile = request.files["Recipes"]
        imageFile = request.files["coverPage"]
        nameRecipes = request.form["NameOfRecipes"]
        TypeOfRecipes = request.form["TypeOfRecipes"]

        key = nameRecipes + ".pdf"
        if TypeOfRecipes == "Breakfast":
            fs = gridfs_bf
        elif TypeOfRecipes == "Lunch":
            fs = gridfs_lu
        elif TypeOfRecipes == "Dinner":
            fs = gridfs_dn
        elif TypeOfRecipes == "Snack":
            fs = gridfs_sn
        else:
            return render_template("update.html", error="Input Correct Type of Recipes")
        try:
            fs.put(recipFile.read(), filename=key)
            fs.put(imageFile.read(), filename=nameRecipes)
            return render_template(
                "update.html", userName=username, message="Upload Successful!"
            )
        except Exception as e:
            app.logger.error("Error connecting to MongoDB:", e)
    return render_template("update.html")


def getFileCount(dbName):
    count = 0
    if dbName == "breakfast":
        count = len(list(gridfs_bf.find()))
    elif dbName == "lunch":
        count = len(list(gridfs_lu.find()))
    elif dbName == "dinner":
        count = len(list(gridfs_dn.find()))
    else:
        count = len(list(gridfs_sn.find()))
    return count


@app.route("/description/<pdf_id>/<tr>", methods=["GET", "POST"])
def description(pdf_id, tr):
    fileURL = url_for("display_pdf", file_id=pdf_id, tr=tr)
    return render_template("description.html", fileURL=fileURL)


@app.route("/display_pdf/<file_id>/<tr>")
def display_pdf(file_id, tr):
    if tr == "Breakfast":
        fs = gridfs_bf
    elif tr == "Lunch":
        fs = gridfs_lu
    elif tr == "Dinner":
        fs = gridfs_dn
    else:
        fs = gridfs_sn
    gridfs_file = fs.get(ObjectId(file_id))
    mimetype = gridfs_file.content_type or "application/pdf"

    response = send_file(
        io.BytesIO(gridfs_file.read()),
        mimetype=mimetype,
        download_name=f"document.pdf",
    )

    return response


@app.route("/display_image/<file_id>/<tr>")
def display_image(file_id, tr):
    if tr == "Breakfast":
        fs = gridfs_bf
    elif tr == "Lunch":
        fs = gridfs_lu
    elif tr == "Dinner":
        fs = gridfs_dn
    else:
        fs = gridfs_sn
    gridfs_file = fs.get(ObjectId(file_id))
    mimetype = gridfs_file.content_type if gridfs_file.content_type else "image/png"

    response = send_file(
        io.BytesIO(gridfs_file.read()),
        mimetype=mimetype,
    )

    return response


@app.route("/breakfast", methods=["GET", "POST"])
def breakfast():
    count = getFileCount("breakfast")
    images = []
    names = []
    pdfs = []
    for i in range(4):
        rb0 = randrange(count)
        random_file0 = gridfs_bf.find()[rb0]
        if random_file0.filename.endswith(".pdf"):
            name0 = random_file0.filename[:-4]  # Remove the file extension
            pdf_file0 = gridfs_bf.find_one({"filename": random_file0.filename})
            image_file0 = gridfs_bf.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)
        else:
            name0 = random_file0.filename
            pdf_file0 = gridfs_bf.find_one({"filename": name0 + ".pdf"})
            image_file0 = gridfs_bf.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)

    return render_template(
        "breakfast.html",
        detailforbreakfast=url_for("description", pdf_id=pdfs[0]._id, tr="Breakfast"),
        detailforlunch=url_for("description", pdf_id=pdfs[1]._id, tr="Breakfast"),
        detailfordinner=url_for("description", pdf_id=pdfs[2]._id, tr="Breakfast"),
        detailforsnack=url_for("description", pdf_id=pdfs[3]._id, tr="Breakfast"),
        picture1=url_for("display_image", file_id=images[0]._id, tr="Breakfast"),
        name1=names[0],
        picture2=url_for("display_image", file_id=images[1]._id, tr="Breakfast"),
        name2=names[1],
        picture3=url_for("display_image", file_id=images[2]._id, tr="Breakfast"),
        name3=names[2],
        picture4=url_for("display_image", file_id=images[3]._id, tr="Breakfast"),
        name4=names[3],
    )


@app.route("/lunch", methods=["GET", "POST"])
def lunch():
    count = getFileCount("lunch")
    images = []
    names = []
    pdfs = []
    for i in range(4):
        rb0 = randrange(count)
        random_file0 = gridfs_lu.find()[rb0]
        if random_file0.filename.endswith(".pdf"):
            name0 = random_file0.filename[:-4]  # Remove the file extension
            pdf_file0 = gridfs_lu.find_one({"filename": random_file0.filename})
            image_file0 = gridfs_lu.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)
        else:
            name0 = random_file0.filename
            pdf_file0 = gridfs_lu.find_one({"filename": name0 + ".pdf"})
            image_file0 = gridfs_lu.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)

    return render_template(
        "lunch.html",
        detailforbreakfast=url_for("description", pdf_id=pdfs[0]._id, tr="Lunch"),
        detailforlunch=url_for("description", pdf_id=pdfs[1]._id, tr="Lunch"),
        detailfordinner=url_for("description", pdf_id=pdfs[2]._id, tr="Lunch"),
        detailforsnack=url_for("description", pdf_id=pdfs[3]._id, tr="Lunch"),
        picture1=url_for("display_image", file_id=images[0]._id, tr="Lunch"),
        name1=names[0],
        picture2=url_for("display_image", file_id=images[1]._id, tr="Lunch"),
        name2=names[1],
        picture3=url_for("display_image", file_id=images[2]._id, tr="Lunch"),
        name3=names[2],
        picture4=url_for("display_image", file_id=images[3]._id, tr="Lunch"),
        name4=names[3],
    )


@app.route("/dinner", methods=["GET", "POST"])
def dinner():
    count = getFileCount("dinner")
    images = []
    names = []
    pdfs = []
    for i in range(4):
        rb0 = randrange(count)
        random_file0 = gridfs_dn.find()[rb0]
        if random_file0.filename.endswith(".pdf"):
            name0 = random_file0.filename[:-4]  # Remove the file extension
            pdf_file0 = gridfs_dn.find_one({"filename": random_file0.filename})
            image_file0 = gridfs_dn.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)
        else:
            name0 = random_file0.filename
            pdf_file0 = gridfs_dn.find_one({"filename": name0 + ".pdf"})
            image_file0 = gridfs_dn.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)

    return render_template(
        "dinner.html",
        detailforbreakfast=url_for("description", pdf_id=pdfs[0]._id, tr="Dinner"),
        detailforlunch=url_for("description", pdf_id=pdfs[1]._id, tr="Dinner"),
        detailfordinner=url_for("description", pdf_id=pdfs[2]._id, tr="Dinner"),
        detailforsnack=url_for("description", pdf_id=pdfs[3]._id, tr="Dinner"),
        picture1=url_for("display_image", file_id=images[0]._id, tr="Dinner"),
        name1=names[0],
        picture2=url_for("display_image", file_id=images[1]._id, tr="Dinner"),
        name2=names[1],
        picture3=url_for("display_image", file_id=images[2]._id, tr="Dinner"),
        name3=names[2],
        picture4=url_for("display_image", file_id=images[3]._id, tr="Dinner"),
        name4=names[3],
    )


@app.route("/snack", methods=["GET", "POST"])
def snack():
    count = getFileCount("snack")
    images = []
    names = []
    pdfs = []
    for i in range(4):
        rb0 = randrange(count)
        random_file0 = gridfs_sn.find()[rb0]
        if random_file0.filename.endswith(".pdf"):
            name0 = random_file0.filename[:-4]  # Remove the file extension
            pdf_file0 = gridfs_sn.find_one({"filename": random_file0.filename})
            image_file0 = gridfs_sn.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)
        else:
            name0 = random_file0.filename
            pdf_file0 = gridfs_sn.find_one({"filename": name0 + ".pdf"})
            image_file0 = gridfs_sn.find_one({"filename": name0})
            images.append(image_file0)
            names.append(name0)
            pdfs.append(pdf_file0)

    return render_template(
        "snack.html",
        detailforbreakfast=url_for("description", pdf_id=pdfs[0]._id, tr="Snack"),
        detailforlunch=url_for("description", pdf_id=pdfs[1]._id, tr="Snack"),
        detailfordinner=url_for("description", pdf_id=pdfs[2]._id, tr="Snack"),
        detailforsnack=url_for("description", pdf_id=pdfs[3]._id, tr="Snack"),
        picture1=url_for("display_image", file_id=images[0]._id, tr="Snack"),
        name1=names[0],
        picture2=url_for("display_image", file_id=images[1]._id, tr="Snack"),
        name2=names[1],
        picture3=url_for("display_image", file_id=images[2]._id, tr="Snack"),
        name3=names[2],
        picture4=url_for("display_image", file_id=images[3]._id, tr="Snack"),
        name4=names[3],
    )


if __name__ == "__main__":
    app.run(debug=True)
