# This is a Python script to test 3D heat mapping
import numpy as np
import math
import matplotlib.pyplot as plt

# importing openCV
import cv2
import pickle as pkl

# importing mediapipe and utilities
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# variables
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# low = red (this is the reference list to define the color of a point based on it's value)
colors_list_rg = [[1, 0, 0], [0.9, 0.4, 0], [0.9, 0.6, 0], [0.85, 0.6, 0], [0.8, 0.7, 0], [0.8, 0.8, 0],
                  [0.8, 0.9, 0], [0.6, 0.95, 0], [0.4, 1, 0], [0, 1, 0]]


# this function (plot_colors) takes in the starting and ending arguments for the x,y,z axis and passes all points in
# that range through a given (optimization) function and then plots by color based on the output of that function
def plot_colors(func, func_max, x_0, x_end, y_0, y_end, z_0, z_end, step=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    rang = func_max/10
    z = z_0
    while z <= z_end:
        y = y_0
        while y <= y_end:
            x = x_0
            while x <= x_end:
                # divide the total range of possible numbers (outputs from the function) by 10 and then divide
                # all outputs by that number and floor it to determine the color
                #####################
                # here the optimization (or any other personal function will be passed in to determine what color
                # in the gradient will be plotted
                out = func(x, y, z) # pass location into the given function, based on the value of the parameter
                if out >= func_max:
                    color = colors_list_rg[9]
                else:
                    color = colors_list_rg[math.floor(out/rang)]

                ax.scatter(x, y, z, color=color)
                x = x + step
            y = y + step
        z = z + step

    plt.show()


def org_dist(x, y, z):
    # returns distance from the origin
    return x + y + z


# code is from Google mediapipe: https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
def draw_image_body(images):
    # Run MediaPipe Pose and draw pose landmarks.
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        for img in images:
            image = cv2.imread(img)
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print nose landmark.
            image_hight, image_width, _ = image.shape
            if not results.pose_landmarks:
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight})'
            )

            # Draw pose landmarks.
            print(f'Pose landmarks of {img}:')
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            resize_and_show(annotated_image)


# code is from Google mediapipe: https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
def plot_body(images):
    with mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        for img in images:
            image = cv2.imread(img)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print the real-world 3D coordinates of nose in meters with the origin at
            # the center between hips.
            print('Nose world landmark:'),
            print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])

            all_joints = [0] * 33
            # 33 joints are used
            x = 0
            while x < 33:
                all_joints[x] = [results.pose_world_landmarks.landmark[x].x, results.pose_world_landmarks.landmark[x].y,
                                 results.pose_world_landmarks.landmark[x].z]
                x = x + 1

            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            # save point as output
            return all_joints


# takes in the desired area size, a function and then plots by color based on the output of that function while also
# plotting (drawing coming soon) body coordinates
def draw_body(body, func, func_max, size, step=1, ind_poi=16):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # this plots the body points
    i = 0
    while i < 33:
        # iterates through list of body indices and co-ords (scaled up by 5)
        curr = body[i]
        ax.scatter(curr[0], curr[1], curr[2], color='pink', s=10)
        i = i + 1

    # this plots the body lines
    # left arm
    x1 = np.array([body[12][0], body[14][0], body[16][0], body[22][0]])
    y1 = np.array([body[12][1], body[14][1], body[16][1], body[22][1]])
    z1 = np.array([body[12][2], body[14][2], body[16][2], body[22][2]])
    # left hand
    x2 = np.array([body[16][0], body[18][0], body[20][0], body[16][0]])
    y2 = np.array([body[16][1], body[18][1], body[20][1], body[16][1]])
    z2 = np.array([body[16][2], body[18][2], body[20][2], body[16][2]])
    # right arm
    x3 = np.array([body[12][0], body[11][0], body[13][0], body[15][0], body[21][0]])
    y3 = np.array([body[12][1], body[11][1], body[13][1], body[15][1], body[21][1]])
    z3 = np.array([body[12][2], body[11][2], body[13][2], body[15][2], body[21][2]])
    # right hand
    x4 = np.array([body[15][0], body[19][0], body[17][0], body[15][0]])
    y4 = np.array([body[15][1], body[19][1], body[17][1], body[15][1]])
    z4 = np.array([body[15][2], body[19][2], body[17][2], body[15][2]])
    # left body + leg + foot
    x5 = np.array([body[12][0], body[24][0], body[26][0], body[28][0], body[30][0], body[32][0], body[28][0]])
    y5 = np.array([body[12][1], body[24][1], body[26][1], body[28][1], body[30][1], body[32][1], body[28][1]])
    z5 = np.array([body[12][2], body[24][2], body[26][2], body[28][2], body[30][2], body[32][2], body[28][2]])
    # waist
    x6 = np.array([body[24][0], body[23][0]])
    y6 = np.array([body[24][1], body[23][1]])
    z6 = np.array([body[24][2], body[23][2]])
    # right body + left + foot
    x7 = np.array([body[11][0], body[23][0], body[25][0], body[27][0], body[29][0], body[31][0], body[27][0]])
    y7 = np.array([body[11][1], body[23][1], body[25][1], body[27][1], body[29][1], body[31][1], body[27][1]])
    z7 = np.array([body[11][2], body[23][2], body[25][2], body[27][2], body[29][2], body[31][2], body[27][2]])

    # plotting lines
    ax.plot(x1, y1, z1, color='black')
    ax.plot(x2, y2, z2, color='black')
    ax.plot(x3, y3, z3, color='black')
    ax.plot(x4, y4, z4, color='black')
    ax.plot(x5, y5, z5, color='black')
    ax.plot(x6, y6, z6, color='black')
    ax.plot(x7, y7, z7, color='black')

    # this plots the heatmap around the point of interest
    poi = body[ind_poi]  # this is the right wrist point by default
    rang = func_max / 10  # determines tiers for color

    # center the point of interest in the box of points
    z = poi[2] - size/2
    print("poi: ", poi)

    # loop through size of area of interest
    while z <= (poi[2] + size/2 + step):
        y = poi[1] - size/2
        while y <= (poi[1] + size/2 + step):
            x = poi[0] - size/2
            while x <= (poi[0] + size/2 + step):
                # the optimization function is utilized to determine color
                out = func(poi, x, y, z)  # pass location into the given function, based on the value of the parameter
                if out >= func_max:
                    color = colors_list_rg[9]
                else:
                    color = colors_list_rg[math.floor(out / rang)]

                ax.scatter(x, y, z, color=color, s=5)
                x = x + step
            y = y + step
        z = z + step

    ax.view_init(elev=280, azim=90)
    plt.show()


# code is from Google mediapipe: https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def build_map_pkl(func_data, points, func_max, func_min, body_pts=None, bp_body_pts=None, ag_body_pts=None, slp_body_pts=None, smpl_ag_body_pts=None, go=True):
    # initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if go:
        # begin iteration
        i = 0
        diff = func_max - func_min
        rang = diff/10
        for point in points:
            # extract the corresponding function value
            value = func_data[i]
            if value > func_max:
                color = colors_list_rg[9]
            else:
                # here we have to subtract the min (as if it's from 0 to func_max - func_min)
                color = colors_list_rg[math.floor((value - diff)/rang)]
            # plot the point
            ax.scatter(point[0], point[1], point[2], color=color, s=5)
            i = i + 1
    # plot the body points then draw them (from Louis ag model)
    if ag_body_pts is not None:
        i = 0
        hip = ag_body_pts[4] # using neck instead // not renaming for time
        for body_point in ag_body_pts:
            body_point = np.array(body_point) - np.array(hip)
            body_point = body_point[0]
            # print("body_point: ", body_point)
            if i == 4:  # do nothing
                ax.scatter(body_point[0], body_point[1], body_point[2], color='green', s=10)
            else:
                ax.scatter(body_point[0], body_point[1], body_point[2], color='blue', s=10)
            i = i + 1

        if True: # this works for the smpl body model 
            # define the body line indexes
            # print("hip: ", hip)
            # hip = hip[0]
            main_body = [4, 1]
            shoulders = [3, 4, 5]
            left_arm = [3, 6, 8] # this works for SMPL body, commented out for testing Henry's model
            right_arm = [5, 7, 9] # same as left arm
            hips = [0, 1, 2]
            left_leg = [0, 10, 12]
            right_leg = [2, 11, 13]
            # draw the main body
            x = build_body_line(ag_body_pts, main_body, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, main_body, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, main_body, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue', label='ag_body')
            # draw shoulders
            x = build_body_line(ag_body_pts, shoulders, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, shoulders, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, shoulders, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw left arm
            x = build_body_line(ag_body_pts, left_arm, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, left_arm, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, left_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw right arm
            x = build_body_line(ag_body_pts, right_arm, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, right_arm, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, right_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw hips
            x = build_body_line(ag_body_pts, hips, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, hips, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, hips, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw left leg
            x = build_body_line(ag_body_pts, left_leg, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, left_leg, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, left_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw right leg
            x = build_body_line(ag_body_pts, right_leg, 0, sub=hip)[0]
            y = build_body_line(ag_body_pts, right_leg, 1, sub=hip)[0]
            z = build_body_line(ag_body_pts, right_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
    
    if smpl_ag_body_pts is not None:
        i = 0
        hip = smpl_ag_body_pts[4] # using neck instead, not renaming for time
        for body_point in smpl_ag_body_pts:
            body_point = np.array(body_point) - np.array(hip)
            # body_point = body_point[0]
            # print("body_point: ", body_point)
            if i == 4:  # do nothing
                ax.scatter(body_point[0], body_point[1], body_point[2], color='green', s=10)
            else:
                ax.scatter(body_point[0], body_point[1], body_point[2], color='black', s=10)
            i = i + 1

        if True: # this works for the smpl body model 
            # define the body line indexes
            # print("hip: ", hip)
            # hip = hip[0]
            main_body = [4, 1]
            shoulders = [3, 4, 5]
            left_arm = [3, 6, 8] # this works for SMPL body, commented out for testing Henry's model
            right_arm = [5, 7, 9, ] # same as left arm
            hips = [0, 1, 2]
            left_leg = [0, 10, 12]
            right_leg = [2, 11, 13]
            # draw the main body
            x = build_body_line(smpl_ag_body_pts, main_body, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, main_body, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, main_body, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black', label='smpl_ag_body')
            # draw shoulders
            x = build_body_line(smpl_ag_body_pts, shoulders, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, shoulders, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, shoulders, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left arm
            x = build_body_line(smpl_ag_body_pts, left_arm, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, left_arm, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, left_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right arm
            x = build_body_line(smpl_ag_body_pts, right_arm, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, right_arm, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, right_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw hips
            x = build_body_line(smpl_ag_body_pts, hips, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, hips, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, hips, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left leg
            x = build_body_line(smpl_ag_body_pts, left_leg, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, left_leg, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, left_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right leg
            x = build_body_line(smpl_ag_body_pts, right_leg, 0, sub=hip)[0]
            y = build_body_line(smpl_ag_body_pts, right_leg, 1, sub=hip)[0]
            z = build_body_line(smpl_ag_body_pts, right_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
    
    if slp_body_pts is not None:
        i = 0
        hip = slp_body_pts[0]
        print("hip: ", hip)
        for body_point in slp_body_pts:
            body_point = body_point - hip
            print("body_point: ", body_point)
            if i == 0:  # do nothing
                ax.scatter(body_point[0], body_point[1], body_point[2], color='green', s=10)
            else:
                ax.scatter(body_point[0], body_point[1], body_point[2], color='pink', s=10)
            i = i + 1

        if True: # this works for the smpl body model 
            # define the body line indexes
            main_body = [12, 0]
            shoulders = [16, 12, 17]
            left_arm = [16, 18, 20, 22] # this works for SMPL body, commented out for testing Henry's model
            right_arm = [17, 19, 21, 23] # same as left arm
            hips = [1, 0, 2]
            left_leg = [1, 4, 7, 10]
            right_leg = [2, 5, 8, 11]
            # draw the main body
            # hip = []
            x = build_body_line(slp_body_pts, main_body, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, main_body, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, main_body, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black', label='slp_smpl_body')
            # draw shoulders
            x = build_body_line(slp_body_pts, shoulders, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, shoulders, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, shoulders, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left arm
            x = build_body_line(slp_body_pts, left_arm, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, left_arm, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, left_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right arm
            x = build_body_line(slp_body_pts, right_arm, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, right_arm, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, right_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw hips
            x = build_body_line(slp_body_pts, hips, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, hips, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, hips, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left leg
            x = build_body_line(slp_body_pts, left_leg, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, left_leg, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, left_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right leg
            x = build_body_line(slp_body_pts, right_leg, 0, sub=hip)[0]
            y = build_body_line(slp_body_pts, right_leg, 1, sub=hip)[0]
            z = build_body_line(slp_body_pts, right_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')    
    
    # plot the body points then draw them (SMPL)
    if body_pts is not None:
        i = 0
        hip = body_pts[0]
        print("hip: ", hip)
        for body_point in body_pts:
            body_point = body_point - hip
            print("body_point: ", body_point)
            if i == 0:  # do nothing
                ax.scatter(body_point[0], body_point[1], body_point[2], color='green', s=10)
            else:
                ax.scatter(body_point[0], body_point[1], body_point[2], color='pink', s=10)
            i = i + 1

        if True: # this works for the smpl body model 
            # define the body line indexes
            main_body = [0, 9, 12, 15]
            shoulders = [16, 12, 17]
            left_arm = [16, 18, 20, 22] # this works for SMPL body, commented out for testing Henry's model
            right_arm = [17, 19, 21, 23] # same as left arm
            hips = [1, 0, 2]
            left_leg = [1, 4, 7, 10]
            right_leg = [2, 5, 8, 11]
            # draw the main body
            # hip = []
            x = build_body_line(body_pts, main_body, 0, sub=hip)[0]
            y = build_body_line(body_pts, main_body, 1, sub=hip)[0]
            z = build_body_line(body_pts, main_body, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black', label='slp_smpl_body')
            # draw shoulders
            x = build_body_line(body_pts, shoulders, 0, sub=hip)[0]
            y = build_body_line(body_pts, shoulders, 1, sub=hip)[0]
            z = build_body_line(body_pts, shoulders, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left arm
            x = build_body_line(body_pts, left_arm, 0, sub=hip)[0]
            y = build_body_line(body_pts, left_arm, 1, sub=hip)[0]
            z = build_body_line(body_pts, left_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right arm
            x = build_body_line(body_pts, right_arm, 0, sub=hip)[0]
            y = build_body_line(body_pts, right_arm, 1, sub=hip)[0]
            z = build_body_line(body_pts, right_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw hips
            x = build_body_line(body_pts, hips, 0, sub=hip)[0]
            y = build_body_line(body_pts, hips, 1, sub=hip)[0]
            z = build_body_line(body_pts, hips, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw left leg
            x = build_body_line(body_pts, left_leg, 0, sub=hip)[0]
            y = build_body_line(body_pts, left_leg, 1, sub=hip)[0]
            z = build_body_line(body_pts, left_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')
            # draw right leg
            x = build_body_line(body_pts, right_leg, 0, sub=hip)[0]
            y = build_body_line(body_pts, right_leg, 1, sub=hip)[0]
            z = build_body_line(body_pts, right_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='black')    
    
    # Henry's model
    if bp_body_pts is not None:
        i = 0
        hip = bp_body_pts[3]
        for body_point in bp_body_pts:
            body_point = body_point - hip
            if i == 0:  # do nothing
                ax.scatter(body_point[0], body_point[1], body_point[2], color='green', s=10)
            else:
                ax.scatter(body_point[0], body_point[1], body_point[2], color='pink', s=10)
            i = i + 1
        if True: # this works for Henry's body pressure data saved within the optimization code
            # define the body line indexes
            main_body = [0, 1, 3]
            # shoulders = [6, 7]
            left_arm = [1, 7, 9, 11] 
            right_arm = [1, 6, 8, 10]
            hips = [16, 3, 17]
            left_leg = [17, 15, 19]
            right_leg = [16, 14, 18]
            # draw the main body
            x = build_body_line(bp_body_pts, main_body, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, main_body, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, main_body, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue', label='result_realtime_handover')
            # draw left arm
            x = build_body_line(bp_body_pts, left_arm, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, left_arm, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, left_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw right arm
            x = build_body_line(bp_body_pts, right_arm, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, right_arm, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, right_arm, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw hips
            x = build_body_line(bp_body_pts, hips, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, hips, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, hips, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw left leg
            x = build_body_line(bp_body_pts, left_leg, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, left_leg, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, left_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')
            # draw right leg
            x = build_body_line(bp_body_pts, right_leg, 0, sub=hip)[0]
            y = build_body_line(bp_body_pts, right_leg, 1, sub=hip)[0]
            z = build_body_line(bp_body_pts, right_leg, 2, sub=hip)[0]
            ax.plot(x, y, z, color='blue')


    ax.view_init(elev=90, azim=0)
    plt.legend(loc="upper left")
    plt.show()

def build_body_line(body_pts, arrs, ind, sub=[]):
    arr = np.zeros(shape = (1, len(arrs)))
    x = 0
    if len(sub) == 0:
        # print("no subtraction factor")
        for i in arrs:
            arr[0][x] = (body_pts[i][ind])
            x = x + 1
    elif len(sub) == 1:
        sub = sub[0]
        for i in arrs:
            arr[0][x] = (body_pts[i][0][ind] - sub[ind])
            x = x + 1
    else:
        for i in arrs:
            arr[0][x] = (body_pts[i][ind] - sub[ind])
            x = x + 1
    return arr


def sort_body(body_data_raw):
    i = 0
    sorted_body = []
    while i < len(body_data_raw):
        add = np.array([body_data_raw[i], body_data_raw[i + 1], body_data_raw[i + 2]])
        sorted_body.append(add)
        i = i + 3
    return sorted_body


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # body_pts = plot_body(["standing_1.jpg"])
    # draw_body(body_pts, point_dist, 0.375, 0.25, 0.125)
    f = open("result_realtime_handover.pkl", 'rb')
    b = open("smpl_body.pkl", 'rb')
    data = pkl.load(f)
    body = pkl.load(b)
    func_out = data["function_array"]
    values = data["points_array"]
    print("max function value: ", max(func_out))
    # body = sort_body(body['human_joints_3D_est'])
    print("almost done")
    build_map_pkl(func_out, values, 341.10947391287306, min(func_out), body_pts=body['human_joints_3D_est'])
    print("done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
