import cv2 as cv


def detect(image_file, video_file):
    # the trained model for people
    people_classifier_file = 'TrainedModel\haarcascade_fullbody.xml'

    # the trained model for cars
    car_classifier_file = 'TrainedModel\pre_trained_model.xml'

    # open the image using opencv
    # image = cv.imread(image_file)

    # read the video by opencv
    video = cv.VideoCapture(video_file)

    # this is the object of classifier
    people_detector = cv.CascadeClassifier(people_classifier_file)

    # this one is to detect the cars
    car_detector = cv.CascadeClassifier(car_classifier_file)

    while True:
        # Read the current frame, the first variable is a boolean for success, and the second one
        # is for the frame itself
        read_successful_bool, frame = video.read()

        if read_successful_bool:
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            break

        # detect the people
        people = people_detector.detectMultiScale(grey_frame)

        # detect the car
        cars = car_detector.detectMultiScale(grey_frame)

        # draw the rectangle around the car
        for [x, y, w, h] in cars:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # draw the rectangle around the people
        for [x, y, w, h] in people:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # show the frame
        cv.imshow("Detecting objects video", frame)

        # this one is to wait for millisecond then moves to the next iteration
        key = cv.waitKey(5)

        if key == 81 or key == 113:
            break

    video.release()
