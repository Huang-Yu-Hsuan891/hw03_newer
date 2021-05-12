# hw03_newer
# first in main.cpp ,用一個while包住rpc call
# 接著，開兩個rpc function，一個是偵測手勢，當按下bottom，會把選到的角度publish到mqtt，另一個則是以選到的角度，如果有偵測到比他大的角度，則publish到mqtt
# 還有，開兩個topic分別是偵測選到的角度和偵測到較大的角度
# 以下，講解兩個rpc分別寫了什麼?
# rpc_function <gesture_ui>
# 在裡面我開了一個thread，我複製了lab8的內容，把while迴圈()，括弧裡的內容設置一個變數，當按下bottom會改變變數，while理所當然地會停下，還有把當下選到的手勢和角度放到ulcd上，這就是我gesture_ui的主要內容
# rpc_function <tilt_angle>
# 在裡面我也開了一個thread，先記錄平放在桌上的加速度值，接著每兩秒在記錄一次加速度的值，看這兩個夾角為多少。
# 而在mqtt方面，開了兩個mqtt_thread，同時也開了兩個publish_message function，同時也開了兩個MQTT::Client，去分別接收兩筆資料
