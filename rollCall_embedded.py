# -*- coding: UTF-8 -*-
import os,dlib,glob,numpy
from skimage import io
import cv2
import imutils
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import pyscreenshot as ImageGrab

'''
if len(sys.argv) != 2:
    print("缺少要辨識的圖片名稱")
    exit()
'''

# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"

# 需要辨識的人臉圖片名稱
#img_path = sys.argv[ 1]

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
def reg():  
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        base = os.path.basename(f)
        # 依序取得圖片檔案人名
        candidate.append(os.path.splitext(base)[ 0])
        img = io.imread(f)
    
        # 1.人臉偵測
        dets = detector(img, 1)
    
        for k, d in enumerate(dets):
            # 2.特徵點偵測
            shape = sp(img, d)
     
            # 3.取得描述子，128維特徵向量
            face_descriptor = facerec.compute_face_descriptor(img, shape)
    
            # 轉換numpy array格式
            v = numpy.array(face_descriptor)
            descriptors.append(v)

reg()
#選擇第一隻攝影機
cap = cv2.VideoCapture(0)
#調整預設影像大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    cv2.imshow("Face Recognition", frame)
    #img = io.imread(img_path)
    
    #push "t"
    if cv2.waitKey(1) == 116:    
        dets = detector(frame, 1)
    
        dist = []
        for k, d in enumerate(dets):
            dist=[]
            shape = sp(frame, d)
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            d_test = numpy.array(face_descriptor)
    
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
        
            frame2 = frame.copy()
            # 以方框標示偵測的人臉
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            # 計算歐式距離
            for i in descriptors:
                dist_ = numpy.linalg.norm(i -d_test)
                dist.append(dist_)
    
            # 將比對人名和比對出來的歐式距離組成一個dict
            c_d = dict(zip(candidate,dist))
    
            # 根據歐式距離由小到大排序
            cd_sorted = sorted(c_d.items(), key = lambda d:d[1])
            print(cd_sorted)
            if cd_sorted[0][1]>0.8:
                name = input("請輸入人名")
                frame2 = imutils.resize(frame2, width = 600)
                cv2.imwrite("rec/"+name+".jpg",frame2)
                reg()
            else:
                # 取得最短距離就為辨識出的人名
                rec_name = cd_sorted[0][0]
                # 將辨識出的人名印到圖片上面
                cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 
                            2, cv2. LINE_AA)
                frame = imutils.resize(frame, width = 600)
            cv2.imshow("Face Recognition", frame)
            cv2.waitKey(0)
            cap.release()   
            cv2.destroyAllWindows()
            break
        ########################################################################
# =============================================================================
#             #measure temperature
#             #量測溫度
#             # 擷取指定範圍畫面（影像大小為 500x500 像素）
#             im = ImageGrab.grab(
#                     bbox=(986,   # X1
#                           133,   # Y1
#                           1361,   # X2
#                           406))  # Y2
#         
#             # 儲存檔案
#             im.save("example4.png")
#             #cv2.waitKey(0)
#             
#         ########################################################################
#         
#         ########################################################################
#         #read temperature
#         #讀取溫度圖檔
#             img = cv2.imread("example4.png")
#         
#             # 裁切區域的 x 與 y 座標（左上角）
#             y = 160
#             #y = 235
#             h = 22
#             '''
#             i==1 x=160 w=15
#             i==2 x=175 w=15
#             i==3 x=190 w=7
#             i==4 x=195 w=20
#             '''
#             for i in range(1,5):
#                 if i==1:
#                     x=153
#                     w=13
#                     # 裁切圖片
#                     crop_img = img[y:y+h:, x:x+w]
#                     cv2.imwrite("test/9.jpg",crop_img)
#                 if i==2:
#                     x=165
#                     w=13
#                     # 裁切圖片
#                     crop_img = img[y:y+h:, x:x+w]
#                     cv2.imwrite("test/2.jpg",crop_img)
#                 if i==3:
#                     x=178
#                     w=5
#                     # 裁切圖片
#                     crop_img = img[y:y+h:, x:x+w]
#                     cv2.imwrite("point.jpg",crop_img)
#                 if i==4:
#                     x=183
#                     w=13
#                     # 裁切圖片
#                     crop_img = img[y:y+h:, x:x+w]
#                     cv2.imwrite("test/3.jpg",crop_img)
#                     
#             #cv2.waitKey(0)
#             
#         ########################################################################
#         
#         ########################################################################
#             #processing temperature
#             #處理溫度圖檔
#             #使用參數方式傳入Training和Test的dataset
#             ap = argparse.ArgumentParser()
#             ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
#             ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
#             args = vars(ap.parse_args())
#         
#             #data用來存放HOG資訊，labels則是存放對應的標籤
#             data = []
#             labels = []
#         
#             #依序讀取training dataset中的圖檔
#             for imagePath in paths.list_images(args["training"]):
#                 
#                     #將資料夾的名稱取出作為該圖檔的標籤
#                     make = imagePath.split("/")[-1]
#         
#                     #—-以下為訓練圖檔的預處理—-#
#                     #載入圖檔,轉為灰階,作模糊化處理
#                     image = cv2.imread(imagePath)
#                     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                     blurred = cv2.GaussianBlur(gray, (3,3), 0)
#         
#                     #作threshold(固定閾值)處理
#                     (T, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
#                     
#                     #使用Canny方法偵側邊緣
#                     edged = imutils.auto_canny(thresh)
#         
#                
#                     #尋找輪廓，只取最大的那個
#                     (image2,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # =============================================================================
# #                     cv2.imshow("Cutted", image2)
# #                     cv2.waitKey(0)
# # =============================================================================
#                     c = max(cnts, key=cv2.contourArea)
#         
#                     #取出輪廓的長寬高，用來裁切原圖檔。
#                     (x, y, w, h) = cv2.boundingRect(c)
#                     Cutted = gray[y:y + h, x:x + w]
#         
#                     #將裁切後的圖檔尺寸更改為60×60。
#                     Cutted = cv2.resize(Cutted, (60, 60))
#                     
#                     #—-訓練圖檔預處理結束—-#
#                     #取得其HOG資訊及視覺化圖檔
#                     (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10), 
#                                                 cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
#         
#                     #將HOG資訊及標籤分別放入data及labels陣列
#                     data.append(H)
#                     labels.append(make)
#         
#             #開始用KNN模型來訓練
#             model = KNeighborsClassifier(n_neighbors=1)
#             
#             #傳入data及labels陣列開始訓練
#             model.fit(data, labels)
#             
#             #準備使用Test Dataset來驗証
#             for (i, imagePath) in enumerate(paths.list_images(args["test"])):
#                 
#                     #從測試資料中讀取圖檔
#                     image = cv2.imread(imagePath)
#         
#                     #—-以下為測試圖檔的預處理—-#
#                     #轉為灰階並模糊化
#                     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# # =============================================================================
# #                     cv2.imshow("Cutted", blurred)
# #                     cv2.waitKey(0)
# # =============================================================================
#                     
#                     #作threshold(固定閾值)處理
#                     (T, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
# # =============================================================================
# #                     cv2.imshow("Cutted", thresh)
# #                     cv2.waitKey(0)
# # =============================================================================
#                     
#                     #使用Canny方法偵側邊緣
#                     edged = imutils.auto_canny(thresh)
# # =============================================================================
# #                     cv2.imshow("Cutted", edged)
# #                     cv2.waitKey(0)
# # =============================================================================
#                     
#                     #尋找輪廓，只取最大的那個
#                     (image2,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     cv2.imshow("Cutted", image2)
#                     #cv2.waitKey(0)
#                     c = max(cnts, key=cv2.contourArea)
#         
#         
#                     #取出輪廓的長寬高，用來裁切原圖檔。
#                     (x, y, w, h) = cv2.boundingRect(c)
#                     Cutted = gray[y:y + h, x:x + w]
#         
#                     #將裁切後的圖檔尺寸更改為60×60。
#                     Cutted = cv2.resize(Cutted, (60, 60))
#                     #cv2.imshow("Cutted", Cutted)
#                     #cv2.waitKey(0)
#                 
#                     #—-訓練圖檔預處理結束—-#
#                     #取得其HOG資訊及視覺化圖檔
#                     (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
#                                                 cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
#         
#                     #使用訓練的模型預測此圖檔
#                     pred = model.predict(H.reshape(1, -1))[0]
#         
#                     #顯示HOG視覺化圖檔
#                     hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#                     hogImage = hogImage.astype("uint8")
#                     #cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
#                     #cv2.waitKey(0)
#         
#                     #將預測數字顯示在圖片上面
#                     #print(pred.title().split("\\")[1])
#                     #print(pred.title().split("\\")[1])
#                     cv2.putText(image, pred.title().split("_")[0], (-2, 21), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
#                 
#                     t_char1=' '
#                     t_char2=' '
#                     t_char3=' '
#                 
#                     if i==0:
#                         t_char1=str(pred.title().split("_")[0])
#                         f_char = t_char1
#                     elif i==1:
#                         t_char2=str(pred.title().split("_")[0])
#                         f_char = f_char +t_char2
#                     elif i==2:
#                         t_char3=str(pred.title().split("_")[0])
#                         f_char = f_char+t_char3
#                         
#                 
# # =============================================================================
# #                     cv2.imshow("{}".format(i + 1), image)
# #                     cv2.waitKey(0)
# # =============================================================================
#             
#             temperature=float(f_char)/10
#             temperatureS=str(temperature)
#             print(temperatureS)
#             
#             cv2.putText(frame, temperatureS, (x1 , y1-45), cv2. FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 
#                             2, cv2. LINE_AA)
#             frame = imutils.resize(frame, width = 600)
#             cv2.imshow("Face Recognition", frame)
#             cv2.waitKey(0)
#             
#             
#         ########################################################################
#         
#         
#         
#             path='./data.txt'
#             f=open(path,'a')
#             name=[rec_name+' '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+temperatureS,'\n']
#             f.writelines(name)
#             f.close()
#             #cv2.waitKey(0)
#         
#             file=open('data.txt','r')
#             data={}
#             try:
#                 for line in file:
#                     data[line.strip()]=1    
#                 for line in iter(data):
#                     print(line)
#             finally:
#                 file.close()
#                 #cv2.waitKey(0)
# =============================================================================
         
    #隨意Key一鍵結束程式
    if cv2.waitKey(1) == 27:
        cap.release()   
        cv2.destroyAllWindows()
        break
