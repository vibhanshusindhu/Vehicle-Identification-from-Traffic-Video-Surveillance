# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:58:47 2020

@author: vibhanshusindhu
"""

import numpy as np
import cv2

# get the video stream
file_video_stream = cv2.VideoCapture('obj_data/obj_videos/06.mp4')

# create a while loop # loop till the last frame of the video
while (file_video_stream.isOpened()):
    # for every iteration obtain the current frame from video stream by using
    # the read function
    ret, current_frame = file_video_stream.read()
    # ret is a bool value which indicates whether the read function was able 
    # to read the video stream correctly or not
    # pass this current_frame to the object detector instead of testing image
    img_to_detect = current_frame
    
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]
    
    # convert the image to blob(binary large object) to pass into model
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416,416),
                                     swapRB = True, crop = False)
    # recommended by yolo authors, scale factor is 0.003922 = 1/255 (255 is the
    # maximum color value of a pixel in the image), width,height of blob is 
    # (320,320). Accepted sizes are (320 x 320), (416 x 416), (609 x 609).More
    # size means more accuracy but less speed. swapRB is set true bcoz originally
    # the colour format in the image is RGB(red,green,blue) but opencv library 
    # converts it to BGR(blue,green,red), so to use it as binary large image we 
    # convert it back to RGB.
    
    # set of 80 class labels(belonging to COCO dataset on which yolo is 
    #                        pretrained)
    class_labels = ["car","truck","motorcycle","bus"]
    
    # declare list of colors as an array
    # green,blue,red,cyan,yellow,purple
    # Split based on ',' and for every split, change type to int
    # convert that to a numpy array to apply color mask to the image numpy array
    
    class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0"]
    # print(class_colors)
    class_colors = [np.array(every_color.split(",")).astype(int) 
                    for every_color in class_colors]
    # print(class_colors)
    class_colors = np.array(class_colors)
    # print(class_colors)
    class_colors = np.tile(class_colors, (1,1))
    # print(class_colors)
    
    # loading pretrained model
    yolo_model = cv2.dnn.readNetFromDarknet('obj_yolov4.cfg',
                                            'obj_yolov4_best.weights')
    
    # Get all layers from the yolo network
    # Loop & find the last layer(output layer) of the yolo network
    yolo_layers = yolo_model.getLayerNames()
    # print(yolo_layers)
    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in 
                         yolo_model.getUnconnectedOutLayers()]
    # print(yolo_output_layer)
    
    # input preprocessed blob into the model & pass through the model
    yolo_model.setInput(img_blob)
    
    # Obtain the detection predictions by the model using forward() method or
    # obtain the detection layers by forwarding through till the output layer
    obj_detection_layers = yolo_model.forward(yolo_output_layer)
    # print(obj_detection_layers)
    
    ######## NMS Change 1 ########
    # initialization for non-maximum supression(NMS)
    # declare list for [class id], [box_centre, width & height], [confidences]
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    ######## NMS Change 1 END ########
    
    # loop over each of the layer outputs
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:
            # object_detection[1 to 4] => will have two centre points, box width   
            # & box height
            # obj_detection[5] => will have scores for all objects within  
            # bounding box
        
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
            
            # take only predictions with confidence more than 20%
            if prediction_confidence > 0.20:
                # get the predicted label
                predicted_class_label = class_labels[predicted_class_id]
                # obtain the bounding box co-oridnates for actual image from 
                # resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, 
                                                                 img_height, 
                                                                 img_width, 
                                                                 img_height])
                (box_centre_x_pt, box_centre_y_pt, box_width, box_height
                ) = bounding_box.astype(int)
                start_x_pt = int(box_centre_x_pt - (box_width/2))
                start_y_pt = int(box_centre_y_pt - (box_height/2))
                
                ############## NMS Change 2 ###############
                # save class id, start x, y, width & height, confidences in a list 
                # for nms processing
                # make sure to pass confidence as float and width and height as 
                # integers
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), 
                                   int(box_height)])
                ############## NMS Change 2 END ###########
                
    ############## NMS Change 3 ###############
    # Applying the NMS will return only the selected max value ids while 
    # suppressing the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threshold 
    # for NMS as 0.4 (adjust and try for better perfomance)
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5,0.4)
    # max_value_ids will be a list of class_ids sorted descendingly in the order 
    # of their confidence_values
    
    # loop through the final set of detections remaining after NMS and draw 
    # bounding box and write text
    my_dict = {} # to keep count of numbers of same class object eg.cars 
    for max_valueid in max_value_ids:
        max_class_id = max_valueid[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        
        #get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
    ############## NMS Change 3 END ###########           
                
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        
        # get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        # print(box_color)
        
        # convert the color numpy array as a list and apply to text & box
        box_color = [int(c) for c in box_color]
        #print(box_color)
        
        count = 1 
        #print(my_dict.keys())
        if predicted_class_label in my_dict.keys():
            count = my_dict[predicted_class_label]
        
        tmp = predicted_class_label
        
        # print the prediction in console
        s = f"{predicted_class_label} {count} : {(prediction_confidence*100):.2f}"
        predicted_class_label = s
        print(f"predicted object {predicted_class_label}")
        
        count+= 1
        my_dict[tmp] = count
        
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), 
                      (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img_to_detect, predicted_class_label, 
                    (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, box_color, 1)
    
    cv2.imshow("Detection Output", img_to_detect)

    # terminate while loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# if keypress detected, release the stream & camera
# releasing the stream & camera
# then destroy/close all opencv windows
file_video_stream.release()
cv2.destroyAllWindows()







