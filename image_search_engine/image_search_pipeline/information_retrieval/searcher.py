# import packages
from .searchresult import SearchResult
from .dist import chi2_distance
import numpy as np
import datetime
import h5py

class Searcher:

    def __init__(self, redisDB, bovwDBPath, featuresDBPath, idf = None,
        distanceMetric = chi2_distance):
        # store the redis database reference, the idf array, and distance
        # metric
        self.redisDB = redisDB
        self.idf = idf
        self.distanceMetric = distanceMetric

        # open both the BOVW database and features database for reading
        self.bovwDB = h5py.File(bovwDBPath, mode = "r")
        self.featuresDB = h5py.File(featuresDBPath, mode = "r")

    def search(self, queryHist, numResults = 10, maxCandidates = 200): # maxCandidates: the maximum number of image indexes to grab from the inverted index that shares a significant number of visual words with the query
        # start the timer to track how long the search took
        startTime = datetime.datetime.now()

        # determine the candidates and sort them in ascending order so they
        # can be read from the BOVW database
        candidateIdxs = self.buildCandidates(queryHist, maxCandidates) # the list of image indexes that shares a significant number of visual words in the query.
        candidateIdxs.sort()

        # grab the histograms for candidates from the BOVW database and
        # initialize the result dictionary
        hists = self.bovwDB["bovw"][candidateIdxs]
        queryHist = queryHist.toarray()
        results = {}

        # if the inverse document frequency array has been supplied, multiply
        # the query by it
        if self.idf is not None:
            queryHist *= self.idf

        # loop over the histograms im BOVW
        for (candidate, hist) in zip(candidateIdxs, hists):
            # if the inverse document frequency array has been supplied, multiply
            # the histogram by it
            if self.idf is not None:
                hist *= self.idf

            # compute the distance between histograms and updated results in dictionary
            d = self.distanceMetric(hist, queryHist)
            results[candidate] = d

        # sort the results, this time replacing the image indexes with the image
        # IDs tehselves
        results = sorted([(v, self.featuresDB["image_ids"][k], k)
            for (k, v) in results.items()])
        results = results[:numResults]

        # return the search results
        return SearchResult(results, (datetime.datetime.now() - startTime).total_seconds())

    def buildCandidates(self, hist, maxCandidates):
        # initialize the redis pipeline
        p = self.redisDB.pipeline()

        # loop over the columns of the (sparse) matrix and create a query to
        # grab all images with an occurrence of the current visual words
        for i in hist.col:
            p.lrange("vw:{}".format(i), 0, -1)

        # execute the pipeline and initialize the candidate list
        pipelineResults = p.execute()
        candidates = []

        # loop over the pipeline results, extract the image index, and update
        # the candidates list
        for results in pipelineResults:
            results = [int(r) for r in results]
            candidates.extend(results)

        # count the occurrence of each of the candidates and sort in descending
        # order
        (imageIdxs, counts) = np.unique(candidates, return_counts = True)
        imageIdxs = [i for (c, i ) in sorted(zip(counts, imageIdxs), reverse = True)]

        # return the image indexes of the candidates
        return imageIdxs[:maxCandidates]

    def finish(self):
        # close the BOVW database and features database
        self.bovwDB.close()
        self.featuresDB.close()
