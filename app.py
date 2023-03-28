from flask import *
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)



def build_model():
    cv=pickle.load(open("cntvect.pkl","rb"))
    df=pickle.load(open("df.pkl","rb"))
    d=df["clean_heading"]
    vectors = cv.transform(d)
    cosine_similarities = cosine_similarity(vectors)
    model={"cosine_similarities": cosine_similarities, "vectorizer":cv, "vectors":vectors}
    return model

def check_article(model,msg):
    df=pickle.load(open("df.pkl","rb"))
    data=model["vectorizer"].transform([msg])
    vectors=model["vectors"]
    query=cosine_similarity(data,vectors)[0]
    indices = query.argsort()[::-1]
    ans= [df["clean_heading"][i] for i in indices[:5]]
    
    return ans
    
    
    
@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/senddata",methods=["POST"])
def fetchdata():
    model=build_model()

    msg=request.form["t1"]
    
    
    ans=check_article(model,msg)
    return render_template("display.html",data=ans)


if (__name__=="__main__"):
    app.run(debug=True)