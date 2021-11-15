import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from imutils.perspective import four_point_transform, order_points

def preprocess(path):

    image = cv2.imread(path)

    grey_conv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    k= np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

    grey=cv2.dilate(grey_conv, k)


    Gaussian_blur = cv2.GaussianBlur(grey, (5,5) , 0)


    threshold=cv2.adaptiveThreshold(Gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)


    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours decreasing order area wise
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros((threshold.shape), np.uint8)
    c = contours[0]


    clone = image.copy()

    perim = cv2.arcLength(c, closed=True)
    poly = cv2.approxPolyDP(c, epsilon=0.02 * perim, closed=True)

    if len(poly) == 4:
        cv2.drawContours(clone, [poly], -1, (0, 0, 255), 2)
        warped = four_point_transform(image, poly.reshape(-1, 2))
    

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    winX = int(warped.shape[1] / 9.0)
    winY = int(warped.shape[0] / 9.0)


    # model = load_model("C:\\Users\\sudha\\OneDrive - North Carolina State University\\Documents\\ALDA\\sudoku-solver\\TMINSTmodel.h5")

    from skimage.segmentation import clear_border
    xrange=range

    labels = []
    centers = []

    predictions = []
    for y in xrange(0, warped.shape[0], winY):
        for x in xrange(0, warped.shape[1], winX):

            window = warped[y : y + winY, x : x + winX]

            if window.shape[0] != winY or window.shape[1] != winX:
                continue

            clone = warped.copy()
            digit = cv2.resize(window, (28, 28))

            _, digit2 = cv2.threshold(digit, 120, 255, cv2.THRESH_BINARY_INV)
            # cv2_imshow(digit2)
            digit3 = clear_border(digit2)
            # cv2_imshow(digit3)
            numpixel = cv2.countNonZero(digit3)
            _, digit4 = cv2.threshold(digit3, 0, 255, cv2.THRESH_BINARY_INV)
            # cv2_imshow(digit4)

            if numpixel < 20:

                label = 0
            else:

                _, digit4 = cv2.threshold(digit4, 0, 255, cv2.THRESH_BINARY_INV)

                digit4 = digit4 / 255.0

                array = model.predict(digit4.reshape(1, 28, 28, 1))

                label = np.argmax(array)

            labels.append(label)
            centers.append(((x + x + winX) // 2, (y + y + winY + 6) // 2))

            cv2.rectangle(clone, (x, y), (x + winX, y + winY), (0, 255, 0), 2)



    grid = np.array(labels).reshape(9, 9)

    zero_indices = zip(*np.where(grid == 0))
    zero_centres = np.array(centers).reshape(9, 9, 2)


    for i in range(9):
        for j in range(9):
            temp = grid[i][j]
            print(temp, end=" ")
        print()

    return grid


M = 9
def puzzle(a):
    for i in range(M):
        for j in range(M):
            print(a[i][j],end = " ")
        print()


def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
             
    for x in range(9):
        if grid[x][col] == num:
            return False
 
 
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True
 
def Suduko(grid, row, col):
 
    if (row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    for num in range(1, M + 1, 1): 
     
        if solve(grid, row, col, num):
         
            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
 



# path = r'images/download.png'
# grid = preprocess(path)


# if (Suduko(grid, 0, 0)):
#     puzzle(grid)
# else:
#     print("Solution does not exist:(")