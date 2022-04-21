import random
from unicodedata import name
from flask import Flask
from flask import request
from flask import redirect
from flask import render_template
from matplotlib.pyplot import text
import json
#靜態路由設定
app = Flask(__name__,
static_folder="homepage", #資料夾名稱 在裡面的都會對應至路徑
static_url_path="/",
template_folder= 'templates',
) #對應的網址路徑 "/"


#動態路由設定#
#內部網頁導向區#
#首頁
@app.route("/") #函式裝飾 設定路由 /對應的處理 
def index():
    # print("請求方法", request.method)
    # print("通訊協定",request.scheme)
    # print("主機名稱",request.host)
    # print("路徑",request.path)
    # print("網址",request.url)
    # print("瀏覽器與os",request.headers.get("user-agent"))
    # print("語言偏好",request.headers.get("accept-language"))
    # print("引薦網址",request.headers.get("referrer")) #從哪連過來
    lang = request.headers.get("accept-language") #瀏覽器的偏好語言
    if(lang.startswith("zh-TW")):
        print("語言偏好:繁中")
        return render_template("originalindex.html",name="桃桃")
    else:
        print("語言偏好:英文or其他")
        return render_template("hello")
        #return redirect("hello")

@app.route("/search")
def search():
    searchinfo = request.args.get("searchinfo","")
    return render_template("search.html",searchinfo=searchinfo)

@app.route("/contact")
def contact():
    return render_template("contact.html",name="桃桃")
#暫時放置區
class TempArea:
    @app.route("/getid")
    def getid():
        id = request.args.get("id",1)
        numberid = int(id)
        print("使用者代號是: ",id)
        # temp =  random.randint(1,100)
        # return "隨機產生num: " + str(temp)

    @app.route("/user/<name>") #名字參數
    def getUser(name):
        if name not in userlist:
            return "you are not a member yet. please register your account"
        return name

    @app.route("/data/<dataname>") #資料
    def getData(dataname):
        if dataname not in userlist:
            return "you are not a member yet. please register your account"
        return dataname

    userlist = ('chloe','aaron','simo','meistu','better')

#啟動
if __name__ == "__main__":
    app.run(port=3000) #啟動server

