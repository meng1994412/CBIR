# import packages
from __future__ import print_function
from image_search_pipeline.descriptors import DetectAndDescribe
from image_search_pipeline.information_retrieval import BagOfVisualWords
from image_search_pipeline.information_retrieval import Searcher
from image_search_pipeline.information_retrieval import SpatialVerifier
from image_search_pipeline import ResultsMontage
from scipy.spatial import distance
from redis import Redis
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import argparse
import pickle
import imutils
import json
import cv2

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory of indexed images")
ap.add_argument("-f", "--features_db", required = True, help = "Path to the features databse")
ap.add_argument("-b", "--bovw_db", required = True, help = "Path to bag-of-visual-words database")
ap.add_argument("-c", "--codebook", required = True, help = "Path to the codebook")
ap.add_argument("-i", "--idf", required = True, help = "Path to the inverted document frequency array")
ap.add_argument("-r", "--relevant", required = True, help = "Path to relevant dictionary")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
args = vars(ap.parse_args())

# initialize the keypoint detector, local invariant descriptor, and descriptor
detector = FeatureDetector_create("SURF")
descriptor = DescriptorExtractor_create("RootSIFT")
dad = DetectAndDescribe(detector, descriptor)

# load the inverted document frequency array and codebook vocabulary
# then initialize the bag-of-visual-words transformer
idf = pickle.loads(open(args["idf"], "rb").read())
vocab = pickle.loads(open(args["codebook"], "rb").read())
bovw = BagOfVisualWords(vocab)

# load the relevant queries dictionary and lookup the relevant rule results for the
# query image
relevant = json.loads(open(args["relevant"]).read())
queryFilename = args["query"][args["query"].rfind("/") + 1:]
queryRelevant = relevant[queryFilename]

# load the query and process it
queryImage = cv2.imread(args["query"])
cv2.imshow("Query", imutils.resize(queryImage, width = 320))
queryImage = imutils.resize(queryImage, width = 320)
queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)

# extract features from the query image and construct bag-of-visual-word from it
(queryKps, queryDescs) = dad.describe(queryImage)
queryHist = bovw.describe(queryDescs).tocoo()

# connect to redis and perform search
redisDB = Redis(host = "localhost", port = 6379, db = 0)
searcher = Searcher(redisDB, args["bovw_db"], args["features_db"], idf = idf,
    distanceMetric = distance.cosine)
sr = searcher.search(queryHist, numResults = 20)
print("[INFO] search took: {:.2f}".format(sr.search_time))

# spatially verified the results
spatialVerifier = SpatialVerifier(args["features_db"], idf, vocab)
sv = spatialVerifier.rerank(queryKps, queryDescs, sr, numResults = 20)
print("[INFO] spatial verification took: {:.2f}s".format(sv.search_time))

# initialize the results montage
montage = ResultsMontage((240, 320), 5, 20)

# loop over the individual results
for (i, (score, resultID, resultIdx)) in enumerate(sv.results):
    # load the result image and display it
    print("[RESULT] {result_num}. {result} - {score:.2f}".format(result_num = i + 1,
        result = resultID, score = score))
    result = cv2.imread("{}/{}".format(args["dataset"], resultID))
    montage.addResult(result, text = "#{}".format(i + 1),
        highlight = resultID in queryRelevant)

# show the output image of results
cv2.imshow("Result", imutils.resize(montage.montage, height = 700))
cv2.waitKey(0)
searcher.finish()
spatialVerifier.finish()
