def detect_people(frame, hog):
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # return the list of results
    return rects, weights
