import numpy as np 
import cv2
import random
from sklearn.neighbors import KNeighborsClassifier


def gen():
    img = np.full((40,185,3), 255, dtype=np.uint8)
    
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXZYabcdefghijklmnopqrstuvwxyz0123456789'
    c1 = random.randint(0, len(chars)-1)
    c2 = random.randint(0, len(chars)-1)
    c3 = random.randint(0, len(chars)-1)
    c4 = random.randint(0, len(chars)-1)
    c5 = random.randint(0, len(chars)-1)
    c6 = random.randint(0, len(chars)-1)

    key = chars[c1]+chars[c2]+chars[c3]+chars[c4]+chars[c5]+chars[c6]
    captcha = chars[c1]+' '+chars[c2]+' '+chars[c3]+' '+chars[c4]+' '+chars[c5]+' '+chars[c6]

    #2  3  7
    font = [2,3,7]
    f1 = random.randint(0, 2)
    f2 = random.randint(0, 2)

    cv2.putText(img, captcha[:5], (15,25), font[f1], 0.75, (0,0,0), 1, 16)
    cv2.putText(img, captcha[5:], (95,25), font[f2], 0.75, (0,0,0), 1, 16)

    for i in range(300):
        color = np.random.randint(95, 240, 3)
        x, y = 0, 0
        x = random.randint(2, 39)
        y = random.randint(2, 184)
        img[x,y] = color

    for i in range(9):
        c1 = random.randint(95, 240)
        c2 = random.randint(95, 240)
        c3 = random.randint(95, 240)
        x = random.randint(2, 180)
        y = random.randint(2, 40)
        lx = random.randint(30, 50)
        ly = random.randint(-50, 50)
        img = cv2.line(img, (x,y), (x+lx,y+ly), (c1,c2,c3), 1)

    for i in range(7):
        c1 = random.randint(95, 240)
        c2 = random.randint(95, 240)
        c3 = random.randint(95, 240)
        x = random.randint(2,200)
        y = random.randint(2,40)
        r = random.randint(1,30)
        img = cv2.circle(img, (x,y), r, (1,c2,c3), 1)

    img  = cv2.resize(img, (0,0), fx=2, fy=2)


    cv2.imshow(key, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(key+'.jpg', img)

    return key

def dip(key):
    img = cv2.imread(key+'.jpg')

    width, height = img.shape[0], img.shape[1]

    img = cv2.fastNlMeansDenoisingColored(img,None,20,80,4,15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # img = cv2.fastNlMeansDenoising(img,None,100,7,21)

    letters = []
    start, end = -1, -1
    found = False
    for i in range(height):
        in_letter = False
        for j in range(width):
            if img[j,i] == 0:
                in_letter = True
                break
        if not found and in_letter and i-start > 15:
            found = True
            start = i
        if found and not in_letter :
            found = False
            end = i
            letters.append([start, end])

    test = []

    for i in letters:
        crop = img[:,i[0]:i[1]]
        crop = cv2.resize(crop,(40,80))
        test.append(crop)
    
    test = np.array(test)

    return letters, test

def loadDataTrain():
    X = []
    y = []
    upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lower = 'abcdefghijklmnopqrstuvwxyz'
    num = '0123456789'
    for i in lower:
        for j in range(1,4):
            y.append(ord(i))
            X.append(cv2.imread('./data/'+i+str(j)+'.jpg',0))

    for i in upper:
        for j in range(4,7):
            y.append(ord(i))
            X.append(cv2.imread('./data/'+i+str(j)+'.jpg',0))
    
    for i in num:
        for j in range(2,5):
            y.append(ord(i))
            X.append(cv2.imread('./data/'+i+str(j)+'.jpg',0))


    X = np.array(X)
    y = np.array(y)

    return X, y


key = 'V2hzhU'
#key = gen()
letters, test = dip(key)
X, y = loadDataTrain()
#print(X.shape)

X_train = X[:, :].reshape(-1,3200)#.astype(np.float32)
X_test = test[:, :].reshape(-1,3200)#.astype(np.float32)
y_train = y#.astype(np.float32)

#print(X_test[0,:])

#knn = cv2.ml.KNearest_create()
#knn.train(X, 0, y)
#kq1, kq2, kq3, kq4 = knn.findNearest(test, 5)
#print(kq2)

# print(y)


neigh = KNeighborsClassifier(n_neighbors=1)
# train
neigh.fit(X_train, y_train)

# predict
results = neigh.predict(X_test)
print(results)

s = ''
for i in results:
    s+=chr(i)

print()
print('Captcha code:    '+key)
print('Captcha predict: '+s)
print()


# #cv2.imshow(key, test[4])
# #cv2.waitKey()
# #cv2.destroyAllWindows()