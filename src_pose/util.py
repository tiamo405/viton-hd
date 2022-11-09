import numpy as np
import math
import cv2

def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights
# draw the body keypoint and lims
def draw_bodypose(img, poses,model_type = 'body_25'):
    stickwidth = 4
    
    limbSeq = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
                   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
                   [0, 15], [15, 17]]
    njoint = 18
    if model_type == 'body_25':    
        njoint = 25
        limbSeq = [[1,8],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[10,11],\
                            [8,12],[12,13],[13,14],[1,0],[0,15],[15,17],[0,16],\
                                [16,18],[14,19],[19,20],[14,21],[11,22],[22,23],[11,24]]
    # colors = [[0, 0, 153], [0, 85, 255], [0, 255, 255], [0, 170, 255], [0, 255, 255], [0, 255, 85], [0, 255, 0], \
    #           [170, 255, 0], [0, 255, 0], [255, 170, 0], [0, 170, 255], [255, 170, 0], [255, 0, 0], [85, 0, 255], \
    #           [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,0], [255,255,85], [255,255,170],\
    #               [255,255,255],[170,255,255],[85,255,255],[0,255,255]]
    colors = [[0, 0, 153], [0, 51, 153], [0, 153, 102], [0, 102, 153], [0, 153, 153], \
            [0, 153, 51], [0, 153, 0], [51, 153, 0], [255,0, 0], [0, 255, 85],\
             [153, 102, 0], [153, 51, 0], [0, 170, 255], [51, 0, 153], [102, 0, 153],\
             [153, 0, 153], [153, 0, 102], [153, 0, 51], [85, 0, 255], [0, 0, 255], \
                [0, 0, 255],[0, 0, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255]]
    for i in range(njoint):
        for n in range(len(poses)):
            pose = poses[n][i]
            if pose[2] <= 0:
                continue
            x, y = pose[:2]
            cv2.circle(img, (int(x), int(y)), 6, colors[i], thickness=-1)
            # cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 

    for pose in poses:
        for limb,color in zip(limbSeq,colors):
            p1 = pose[limb[0]]
            p2 = pose[limb[1]]
            if p1[2] <=0 or p2[2] <= 0:
                continue
            cur_canvas = img.copy()
            X = [p1[1],p2[1]]
            Y = [p1[0],p2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            img = cv2.addWeighted(img, 0.1, cur_canvas, 0.9, 0)
   
    return img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad
def draw_handpose(canvas, all_hand_peaks, show_number=False) :
    colors = [[0, 0, 60], [0, 0, 90], [0, 0, 120], [0, 0, 153], [0, 60, 60], \
            [0, 90, 90], [0, 120, 120], [0, 153, 153], [30,60, 0], [45, 90, 0],\
             [60, 120, 0], [75, 153, 0], [60, 30, 255], [90, 45, 0], [120, 60, 0],\
             [153, 75, 0], [60, 0, 60], [90, 0, 90], [120, 0, 120], [153, 0, 153]]
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks in all_hand_peaks:
        for limb,color in zip(edges,colors):
            cur_canvas = canvas.copy()
            Y = peaks[limb, 0]
            X = peaks[limb, 1]
            if 0 in Y or 0 in X :
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.1, cur_canvas, 0.9, 0)
        for i in range(20):
            x,y = peaks[i]
            if 0 in peaks[i]:
                continue
            cv2.circle(canvas, (int(x), int(y)), 2, colors[i], thickness=5)
    return canvas
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j
