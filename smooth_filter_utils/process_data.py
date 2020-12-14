import json
import cv2
import numpy as np

def json_back_to_boxes(vid_dir, det_name):

    total_forceps_boxes = []
    with open(vid_dir+det_name) as json_file:
        data = json.load(json_file)
        for frame, frame_detections in data.items():
            forceps_boxes = []
            for detections in frame_detections:
                for key, value in detections.items():
                    if key == "toolbox0.0":
                        forceps_boxes.append(value)

            total_forceps_boxes.append(forceps_boxes)

    return total_forceps_boxes



#THIS CHUNK OF CODE IS USED TO APPLY MAJORITY FILTER
#FILTER PARAMETERS
def filter_box(box, frame_number, total_pred_boxes, f_parm, d_parm, final_frame):
    positives = 0
    frames_checked = 0
    for f in range(max(0, frame_number-f_parm), min(final_frame, frame_number+f_parm, len(total_pred_boxes))): #length of frames
        frames_checked += 1
        for detection in total_pred_boxes[f]:
            distance = abs(detection[0] - box[0]) + abs(detection[1] - box[1]) + abs(detection[2] - box[2]) + abs(detection[3] - box[3])
            if  distance <= d_parm:
                positives += 1
    return float(positives/frames_checked)*100



def NMS_boxes(boxes, NMS_TOL):
    #the purpose of this function of to accept an array of boxes, 
    #and return a subarray removing any boxes which are totally encapsulated
    TOL = 75
    non_supressed_boxes = []
    for box_no, box1 in enumerate(boxes): #checking to see if we can remove box_1
        suppressed = False
        for box2 in boxes:
            if box1[0] + TOL > box2[0] and box1[1] + TOL > box2[1] and box1[2] - TOL < box2[2] and box1[3] - TOL < box2[3] and not np.array_equal(box1, box2):
                suppressed = True
        if suppressed == False:
            non_supressed_boxes.append(box1)
    return non_supressed_boxes



def filt(cur_video, total_tool_boxes):

    print("Filtering detection!")

    video = cv2.VideoCapture(cur_video)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    final_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    #one 20th average dimension * 4
    d_parm = (width+height)/2*.05*4
    print("d_parm is", d_parm)

    NMS_TOL = (width+height)/2 * .2
    print("NMS_TOL is", NMS_TOL)

    f_parm_tools = 3 #50

    new_total_tool_boxes = []

    kept_detections = 0

    for frame_number, frame_boxes in enumerate(total_tool_boxes):

        print('Filtering frame number', frame_number, end = "\r")
        frame_toolbox_detections = []

        total_tool_boxes[frame_number] = NMS_boxes(total_tool_boxes[frame_number], NMS_TOL)

        if len(total_tool_boxes[frame_number]) > 0:
            for detection_number, box in enumerate(total_tool_boxes[frame_number]):
                pos_percentage = filter_box(box, frame_number, total_tool_boxes, f_parm_tools, d_parm, final_frame)
                if pos_percentage > 10:# 48:
                    frame_toolbox_detections.append(box)

        new_total_tool_boxes.append(frame_toolbox_detections)

    return new_total_tool_boxes



#calculates a wierd distance between two boxes
def box_distance(box1, box2):
    return sum(abs(box1[i] - box2[i]) for i in range(len(box1)))


#calculates sum of triangle distance between poses
def pose_distance(pose1, pose2):
    return sum([  ((pose1[i][0]-pose2[i][0])**2 + (pose1[i][1]-pose2[i][1])**2)**0.5 for i in range(len(pose1))   ])


#exponentially weighted point average
def exp_weight_box(cur_box, old_box, alpha):
    return [alpha*cur_box[i] + (1-alpha)*old_box[i] for i in range(len(cur_box))]


#exponentially weighted point average
def exp_weight_pose(cur_pose, old_pose, alpha):
    return [[alpha*cur_pose[i][0] + (1-alpha)*old_pose[i][0], alpha*cur_pose[i][1] + (1-alpha)*old_pose[i][1]] for i in range(len(cur_pose))]


#main smoothing function, tries to match boxes/poses to previous frame, then applies EMA to previous closest match
def smooth(cur_video, total_tool_boxes):
    
    close = 15 #how close does a match need to be? IN NEXT VERSION, SHOULD BE FUNCTION OF VIDEO FRAME SIZE
    alpha = .4 #weight term for EMA
    
    new_total_tool_boxes = [[]]

    for frame_number, frame_boxes in enumerate(total_tool_boxes):
        
        print('Smoothing frame number', frame_number, end = "\r")
        
        frame_toolbox_detections = []

        if len(total_tool_boxes[frame_number]) > 0: #not sure if this if statement does anything    
            for detection_number, box in enumerate(total_tool_boxes[frame_number]): 
                
                closest_distance = 10**8
                
                for last_box_no, last_box in enumerate(new_total_tool_boxes[frame_number-1]):
                    distance = box_distance(box, last_box)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_box = last_box
                if closest_distance < close*17/4:
                    frame_toolbox_detections.append(exp_weight_box(box, closest_box, alpha))
                    
                else:
                    frame_toolbox_detections.append(box)

                    
                    
                    
        new_total_tool_boxes.append(frame_toolbox_detections)
    
    
    #don't take the first frame! that was just a buffer!
    return new_total_tool_boxes[1:]



def annotate_videos(cur_video, out_video, new_total_tool_boxes):
        
    #input movie to overlay predictions on
    video = cv2.VideoCapture(cur_video)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    line_thickness = min(int(height/200), int(width/200))
    print("Line thickness is {} \n".format(line_thickness))
    print("Video is {} FPS\n".format(fps))

    fourcc = cv2.VideoWriter_fourcc('H','2','6','4')

    video_tracked = cv2.VideoWriter(out_video, fourcc, fps, (int(width), int(height)))

    #used to skip ahead in the video

    frame_num = 0

    while video.isOpened() and frame_num < len(new_total_tool_boxes):        
        tool_boxes = new_total_tool_boxes[frame_num]
        #print("TOOL BOXES", tool_boxes)
        print("Annotating Frame Number ", frame_num, end='\r')

        _, frame = video.read() 

        
        img = frame
        image_debug = img.copy()
        image_pose = img.copy()
        
        for it, box in enumerate(tool_boxes):
            if len(box) > 0:
                #print("Annotating", (int(box[0]), int(box[1])), (int(box[2]), int(box[3])))
                cv2.rectangle(image_debug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 0, 0),
                              thickness=line_thickness)

        video_tracked.write(image_debug)
        
        frame_num += 1

    video.release()
    video_tracked.release()
    print("Released video", out_video)
    
    return fps
