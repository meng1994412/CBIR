# import packages
import numpy as np

class DetectAndDescribe:

    def __init__(self, detector, descriptor):
        # store the keypoint detector and local invariant descriptor
        self.detector = detector
        self.descriptor = descriptor

    def describe(self, image, useKpList = True):
        # detect keypoints in the image and extract local invariant descriptor
        kps = self.detector.detect(image)
        (kps, descs) = self.descriptor.compute(image, kps)

        # if there are no keyponts or descriptors, return None
        if len(kps) == 0:
            return (None, None)

        # check to see if the keypoint should be converted to NumPy array
        if useKpList:
            kps = np.int0([kp.pt for kp in kps])

        # return a tuple of the keypoints and descriptors
        return (kps, descs)
        
