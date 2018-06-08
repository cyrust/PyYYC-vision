# -*- coding: utf-8 -*-
"""
Created on Sat May 12 11:52:40 2018

@author: siyavash
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

print(cv2.__version__)

# DESCRIPTION: plots an image using matplotlib
# plotimg: numpy array image to plot
# color: set to True if image is in color, set to False if grayscale
def plot_img(plotimg, color=True):
    plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    if color == True:
        img = cv2.cvtColor(plotimg, cv2.COLOR_BGR2RGB)
        plt.imshow(img, interpolation = 'bicubic')
    else:
        plt.imshow(plotimg, interpolation = 'bicubic', cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


# create a test image
img_w = 600
img_h = img_w
img = np.ones((img_w, img_h, 3), dtype=np.uint8)*255

# square size
square_width = 50
# colors (BGR)
color1 = (0,150,0)
color2 = (0,0,200)
# board location and size
board_x = 100
board_width = 400


# DESCRIPTION: draw a square checker board
# img: img to draw board on
# board_x: x coordinate of the board in img
# board_width: width of the board
# square_width: width of a square cell of the board
# color1: color of one type of square cell on the board
# color2: color of another type of square cell on the board
def draw_board(img, board_x, board_width, square_width, color1, color2):
    
    n = board_width//square_width
    w = square_width

    for i in range(0,n): # cycle through 1 to 7
        for j in range(0,n):
            if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
                color = color1
            else:
                color = color2
            
            # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
            cv2.rectangle(img, (board_x+i*w,board_x+j*w),(board_x+i*w+w,board_x+j*w+w),color,-1)
    return 0

draw_board(img, board_x, board_width, square_width, color1, color2)

#%%
# create a title
def write_title(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 4
    text = "PyYYC Demo"
    #cv.getTextSize(textString, font)-> (textSize, baseline)
    Size, textSize = cv2.getTextSize(text, font,
                                fontScale, fontThickness)
    #cv.putText(img, text, org, font, color) → None
    cv2.putText(img,text, (img_w//2 - Size[0]//2, (img_h-board_width)//4 + Size[1]//2), 
                font, fontScale, (0,0,0), fontThickness)
                
write_title(img)

# DESCRIPTION: create an image of a checker piece
# square_width: width of the image for the checker
# color1: color used for the checker
# color2: color used for the checker
def draw_checker(square_width, color1, color2):
    # create a checker image
    checker = np.ones((square_width,square_width,3), dtype=np.uint8)*255
    #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → None
    cv2.circle(checker, (square_width//2, square_width//2), square_width//2, color1, -1)
    cv2.circle(checker, (square_width//2, square_width//2), square_width//3, color2, -1)
    cv2.circle(checker, (square_width//2, square_width//2), square_width//4, color1, -1)
    return checker

brown1 = np.subtract((47,62,139),(47,50,50))
brown1 = brown1.tolist()
brown2 = np.add(brown1,(40,40,40))
brown2 = brown2.tolist()
checker_brn = draw_checker(square_width, brown1, brown2)

# DESCRIPTION: place checker on the board graphically
# img: image to draw checker on
# checker: image of the checker
# i: the horizontal index of the square to draw checker on
# j: the vertical index of the square to draw checker on
# board_x: x-coordinate of the board
# board_y: y-coordinate of the board
# square_width: size of the board square
# method: if 1 copy checker over, if 2 use weighted addition, if 3 use threholding, if 4 use color filtering
def place_checker(img, checker, i, j, board_x, board_y, square_width, method=1):

    square = img[(board_x+(i-1)*square_width):(board_x+(i)*square_width),
                     (board_y+(j-1)*square_width):(board_y+(j)*square_width)]
                
    if method == 1:
        # simple image placement
        result=checker
        
    elif method == 2:
        # weighted image addition
        #cv.AddWeighted(src1, alpha, src2, beta, gamma, dst) → None
        # dst = src1 * alpha + src2 * beta + gamma
        result = cv2.addWeighted(square, 0.5, checker, 0.5, 0)
        
    elif method == 3:
        # thresholding

        # convert to gray scale
        #cv.cvtColor(src, dst, code) → None
        checkergray = cv2.cvtColor(checker,cv2.COLOR_BGR2GRAY)
        # find brown2 grey color conversion
        brown2pix = np.zeros((1,1,3), dtype=np.uint8)
        brown2pix[0,0] = brown2
        #cv.cvtColor(src, dst, code) → None
        brown2_gry = cv2.cvtColor(brown2pix,cv2.COLOR_BGR2GRAY)
        # threshold checker after graying it with brown2pix as the threshold
        #cv.threshold(src, dst, threshold, maxValue, thresholdType) → None
        r, mask = cv2.threshold(checkergray, brown2_gry[0,0]+5, 255, cv2.THRESH_BINARY_INV)
        # find the inverse of the threshold
        #cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
        mask_inv = cv2.bitwise_not(mask)
        
        # black out the area of checker in square
        #cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
        square_mskd = cv2.bitwise_and(square,square,mask = mask_inv)
        # black out the area of square in checker
        #cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
        checker_mskd = cv2.bitwise_and(checker,checker,mask = mask)
        #cv2.add(src1, src2[, dst[, mask[, dtype]]]) → dst
        result = cv2.add(square_mskd,checker_mskd)
    else:
        # color filtering
        
        #https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
        #cv.cvtColor(src, dst, code) → None
        checker_hsv = cv2.cvtColor(checker, cv2.COLOR_BGR2HSV)
        # find brown1 hsv color conversion
        brown1pix = np.zeros((1,1,3), dtype=np.uint8)
        brown1pix[0,0] = brown1
        #cv.cvtColor(src, dst, code) → None
        brown1_hsv = cv2.cvtColor(brown1pix,cv2.COLOR_BGR2HSV)
        # find brown2 hsv color conversion
        brown2pix = np.zeros((1,1,3), dtype=np.uint8)
        brown2pix[0,0] = brown2
        #cv.cvtColor(src, dst, code) → None
        brown2_hsv = cv2.cvtColor(brown2pix,cv2.COLOR_BGR2HSV)
        
        """         color1=np.array([min(brown1_hsv[0][0][0],brown2_hsv[0][0][0]),
                         min(brown1_hsv[0][0][1],brown2_hsv[0][0][1]),
                         min(brown1_hsv[0][0][2],brown2_hsv[0][0][2])]) #([4+4,255,89-5]) #([4,255,89])
        color2=np.array([max(brown1_hsv[0][0][0],brown2_hsv[0][0][0]),
                         max(brown1_hsv[0][0][1],brown2_hsv[0][0][1]),
                         max(brown1_hsv[0][0][2],brown2_hsv[0][0][2])]) #([4+4,255,89-5]) #([4,255,89])
        """        
        # cv2.inRange(src, lowerb, upperb[, dst]) → dst
        mask = cv2.inRange(checker_hsv, (0,0,0,), (180,255,253)) #brown1_hsv[0][0], brown2_hsv[0][0])
        mask_inv = cv2.bitwise_not(mask)
        # black out the area of checker in square
        #cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
        square_mskd = cv2.bitwise_and(square,square,mask = mask_inv)
        # black out the area of square in checker
        #cv2.bitwise_and(src1, src2[, dst[, mask]]) → dst
        checker_mskd = cv2.bitwise_and(checker,checker,mask = mask)
        #cv2.add(src1, src2[, dst[, mask[, dtype]]]) → dst
        result = cv2.add(square_mskd,checker_mskd)

    img[(board_y+(j-1)*square_width):(board_y+(j)*square_width),
        (board_x+(i-1)*square_width):(board_x+(i)*square_width)]=result
        
# simple placement
place_checker(img,checker_brn, 2,2, board_x, board_x, square_width)
# weighted placement
place_checker(img,checker_brn, 4,2, board_x, board_x, square_width,2)
# filter by thresholding
place_checker(img,checker_brn, 6,2, board_x, board_x, square_width,3)
# filter by color
place_checker(img,checker_brn, 8,2, board_x, board_x, square_width,4)

plot_img(img)

#%%
# DESCRIPTION: perform template matching to find checkers
# img: image to find template in
# template: template image to find
# thredhold: level matching accuracy desired
# imgdraw: image to draw rectangles of the matched templates on
# tmplt_mask: mask to apply to template image before matching for transparency
# rect_color: color of the rectangles to draw in imgdraw of the matches
def find_image(img, template, threshold, imgdraw = None, tmplt_mask = None, rect_color = (0,0,0)):
    # convert inputs into grayscale if color
    if (len(img.shape)==3):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img2 = np.copy(img)
    if (len(template.shape)==3):
        template2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template2 = np.copy(template)
    if tmplt_mask is not None:
        if (len(tmplt_mask.shape)==3):
            tmplt_mask2 = cv2.cvtColor(tmplt_mask, cv2.COLOR_BGR2GRAY)
        else:
            tmplt_mask2 = np.copy(tmplt_mask)
    w,h = template2.shape
    
    # create a full mask if one is not provided
    if tmplt_mask is None:       
        one_img = np.ones((w, h), dtype=np.uint8)
        tmplt_mask = one_img

    # make all variables 32-bit floats
    tmplt_mask2 = np.float32(tmplt_mask)
    template2 = np.float32(template2)
    img2 = np.float32(img2)

    #cv2.matchTemplate(image, templ, method[, result]) → result
    #method: TM_CCOEFF_NORMED (filter for maximum), TM_SQDIFF_NORMED (filter for minimum)
    #https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
    match = cv2.matchTemplate(img2,template2,cv2.TM_SQDIFF, mask=tmplt_mask2)
    match = cv2.normalize(match,match,0,1,cv2.NORM_MINMAX)

    """ 
    match = cv2.matchTemplate(img2,template2,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    print (min_val, " ", min_loc, " ", max_val, " ", max_loc) """

    loc = np.where( match <= threshold)
    
    if imgdraw is not None:
        # draw images found
    
        for pt in zip(*loc[::-1]):
            cv2.rectangle(imgdraw, pt, (pt[0] + h, pt[1] + w), rect_color, 3)
    
    return match


img_searched = np.copy(img)
find_image(img, checker_brn, 0.1, img_searched)

plot_img(img_searched)

#%%
# try canny edge detection
#cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges
edges_img = cv2.Canny(img,10,20)
edges_checker = cv2.Canny(checker_brn,10,20)

plot_img(edges_checker, False)
plot_img(edges_img, False)

img_searched = np.copy(img)
find_image(edges_img, edges_checker, 0.1, img_searched)
plot_img(img_searched)


#%%
# detect board using edge detection algorithm by creating a matching board template image

# perform edge detection on board without anything on it
border = 10
img_board = np.ones((board_width+ border, board_width+ border, 3), dtype=np.uint8)*255  # image to contain only board
draw_board(img_board, border//2, board_width, square_width, color1, color2)  # draw plain board
#cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges
edges_tmplt = cv2.Canny(img_board,10,20)
plot_img(edges_tmplt, False)
#cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) → dst
edges_tmplt_mask = cv2.normalize(edges_tmplt,None, alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

# perform template matching to detect board
find_image(edges_img, edges_tmplt, 0.1, img_searched, edges_tmplt_mask)
plot_img(img_searched)


# make an actual checkers board
img_board_title = np.ones((img_w, img_h, 3), dtype=np.uint8)*255
draw_board(img_board_title, board_x, board_width, square_width, color1, color2)
write_title(img_board_title)


# create checker of different color
cream1 = (135,184,222)
cream2 = (cream1[0]-30,cream1[1]-30,cream1[2]-30)
checker_crm = draw_checker(square_width,cream1,cream2)
# create a board state
n = board_width//square_width
state = [[0 for x in range(n)] for y in range(n)] 

# set up brown
for i in range(0,8):
    for j in range(0,3):
        state[i][j]=0
        if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
            state[i][j] = 1
# set up white
for i in range(0,8):
    for j in range(5,8):
        state[i][j]=0
        if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
            state[i][j] = 2

# function to draw board for a given state
def draw_game_state(state, img):
    img_final = np.copy(img)
    # set up brown
    for i in range(0,8):
        for j in range(0,8):
            if state[i][j]==1:
                place_checker(img_final, checker_brn, i+1, j+1, board_x, board_x, square_width,4)
    # set up white
    for i in range(0,8):
        for j in range(0,8):
            if state[i][j]==2:
                place_checker(img_final, checker_crm, i+1, j+1, board_x, board_x, square_width,4)
    return img_final

# draw game
img_game = draw_game_state(state, img_board_title)
plot_img(img_game)




# get game state from image
def read_state(img_game):
    img_searched = np.copy(img_game)
    edges_img_game = cv2.Canny(img_game,10,20)
    # perform template matching to detect board
    board_match = find_image(edges_img_game, edges_tmplt, 0.1, img_searched, edges_tmplt_mask)
    # predict board location
    min_loc= cv2.minMaxLoc(board_match)[2]
    pred_board_x = min_loc[0]+border//2
    pred_board_y = min_loc[0]+border//2

    # perform template matching to detect checkers
    checker_hsv = cv2.cvtColor(checker_brn, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(checker_hsv, (0,0,0), (180,255,253))
    mask = cv2.normalize(mask,None, alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    checker_brn_match = find_image(img_game, checker_brn, 0.0, img_searched,
        tmplt_mask=mask, rect_color = (0,255,100))
    checker_crm_match = find_image(img_game, checker_crm, 0.0, img_searched,
        tmplt_mask=mask, rect_color = (255,0,0))

    # plot search results
    plot_img(img_searched)
    
    # convert image findings to game state
    # create game state
    n = board_width//square_width
    state = [[0 for x in range(n)] for y in range(n)] 
    
    # np.savetxt("checker_brn_match.csv",checker_brn_match,delimiter=",")

    for i in range(0,8):
        for j in range(0,8):
            # calculate board square locations
            if ((checker_brn_match[(pred_board_y+(j)*square_width):(pred_board_y+(j+1)*square_width),
                (pred_board_x+(i)*square_width):(pred_board_x+(i+1)*square_width)]==0).sum()>0):
                state[i][j]=1
            elif ((checker_crm_match[(pred_board_y+(j)*square_width):(pred_board_y+(j+1)*square_width),
                (pred_board_x+(i)*square_width):(pred_board_x+(i+1)*square_width)]==0).sum()>0):
                state[i][j]=2
            else:
                state[i][j]=0
    return state

# show mask for checkers
checker_hsv = cv2.cvtColor(checker_brn, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(checker_hsv, (0,0,0), (180,255,253))
mask = cv2.normalize(mask,None, alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
plot_img(mask,False)

# draw read state with a move
state = read_state(img_game)
state[2][2]=0
state[3][3]=1
# draw game
img_game2 = draw_game_state(state, img_board_title)
plot_img(img_game2)


# what's changed in the board?
state = read_state(img_game2)


# the end
