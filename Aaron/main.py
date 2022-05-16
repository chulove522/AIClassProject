from flask import *
import pymongo 



#靜態路由設定
app = Flask(__name__,
static_folder="homepage", #資料夾名稱 在裡面的都會對應至路徑
static_url_path="/",      #對應的網址路徑 "/"
template_folder= 'templates',
)
#key設定
app.secret_key = "the key"

#啟動
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80) #啟動server

