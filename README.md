# Content-Based Image Retrieval Syetem
## Project Objectives
* Extracted keypoint detectors and local invariant descriptors of each image in the dataset and stored them in HDF5.
* Clustered the extracted features in HDF5 to form a codebook (resulting centroids of each clustered futures) and visualized each codeword (the centroid) inside the codebook.
* Constructed a bag-of-visual-words (BOVW) representation for each image by quantizing the associated feature vectors into histogram using the codebook created.
* Accepted a query image from the user, constructed the BOVW representation for the query, and performed the actual search.
* Implemented term frequency-inverse document frequency and spatial verification to improve the accuracy of the system.

## Software/Package Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.1/) 3.4
* [Imutils](https://github.com/jrosebr1/imutils)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [HDF5](https://www.h5py.org/)
* [redis](https://redis.io/)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/scipylib/index.html)

## Algorithms & Methods Involved
* Keypoints and descriptors extraction
  * Fast Hessian keypoint detector algorithms
  * Local scale-invariant feature descriptors (RootSIFT)
* Feature storage and indexing
  * Structure HDF5 dataset
* Clustering features to generate a codebook
  * K-means algorithms
* Visualizing codeword entries (centroids of clustered features)
* Vector quantization
  * BOVW extraction
  * BOVW storage and indexing
* Inverted indexing
  * Implement redis for inverted index
* Search performing
* System accuracy evaluation
  * "Points-based" metric
* Term frequency-inverse document frequency (tf-idf)
* Spatial verification (Future Plan)
  * Random Sample Consensus (RANSAC)

## Approaches
* The dataset is about 1000 images from [UKBench](https://archive.org/details/ukbench) dataset.
* The figure below shows the CBIR search pipelines.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/cbir_searching.jpg" width="600">

## Results
### Extract keypoints and descriptors
**This is the step 1 in building the bag of visual word (BOVW).**

In order to extract features from each image in the dataset, I use `Fast Hessian` method for keypoint detectors and use `RootSIFT` for local invariant descriptors.

The `descriptors/` directory (inside `image_search_engine/image_search_pipeline/` directory) contains `detectanddescribe.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/descriptors/detectanddescribe.py)), which implements to extract keypoints and local invariant descriptors from the dataset.

The `index/` directory inside `image_search_engine/image_search_pipeline/` directory contains object-oriented interfaces to the HDF5 dataset to store features. In this part, `baseindexer.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/indexer/baseindexer.py)) and `featureindexer.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/indexer/featureindexer.py)) are used for storing features.

The `index_fetures.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/index_features.py)) is the driver script for gluing all pieces mentioned above. After running this driver script, I have the `features.hdf5` file shown below, which has about 556 MB.

Using the following command line will run the `index_features.py` driver script.

```
python index_features.py --dataset ukbench --features_db output/features.hdf5
```

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/hdf5_database.png" width="200">

Figure 1: `features.hdf5` file, which contains all the features extracted from the whole dataset.

The Figure 2 shows a sample of interior structure inside `features.hdf5` file. I use `HDF5` because of the ease of interaction with the data. We can store huge amounts of data in our `HDF5` dataset and manipulate the data using NumPy. In addition, the `HDF5` format is standardized, meaning that datasets stored in `HDF5` format are inherently portable and can be accessed by other developers using different programming languages, such as C, MATLAB, and Java.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/hdf_database_layout.png" width="500">

Figure 2: A sample of interior structure of the `features.hdf5`.

The `image_ids` dataset has shape (M,) where M is total number of examples in dataset (in this case, M = 1000). And `image_ids` is corresponding to the filename.

The `index` dataset has shape (M, 2) and stores two integers, indicating indexes into `features` dataset for image i.

The `features` dataset has shape (N, 130), where N is the total number of feature vectors extracted from M images in the dataset (in this case, N = 523,505). First two columns are the (x, y)-coordinates of the keypoint associated with the feature vector. The other 128 columns are from `RootSIFT` feature vectors.

### Cluster features
**This is the step 2 in building the bag of visual word (BOVW).**

The next step is to cluster extracted feature vectors to form "vocabulary", or simply result the cluster centers generated by the K-means algorithm.

**Concept of bag of the visual word**

The goal is to take an image that is represented using multiple feature vectors and then construct a histogram for each image of image patch occurrences that tabulate the frequency of each of these prototype vectors. A "prototype" vector is simply a "visual word" — it’s an abstract quantification of a region in an image. Some visual words may encode for corner regions. Others visual words may represent edges. Even other visual words symbolize areas of low texture. Some sample examples of the "visual word" will be demonstrated in next part.

The `Vocabulary` class inside `vocabulary.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/information_retrieval/vocabulary.py)) from `information_retrieval/` directory (inside `image_search_engine/image_search_pipeline/` directory) is used to ingest `features.hdf5` dataset of features and then return cluster centers of visual words. These visual words will serve as our vector prototypes when I quantize the feature vectors into a single histogram of visual word occurrences in one of the following step.

The `cluster_fetures.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/cluster_features.py)) is the driver script that clusters features.

The `MiniBatchKMeans` is used, which is a more efficient and scalable version of the original k-means algorithm. It essentially works by breaking the dataset into small segments, clustering each of the segments individually, then merging the clusters resulting from each of these segments together to form the final solution. This is in stark contrast to the standard k-means algorithm which clusters all of the data in a single segment. While the clusters obtained from mini-batch k-means aren’t necessarily as accurate as the ones using the standard k-means algorithm, the primary benefit is that mini-batch k-means is that it’s often an order of magnitude (or more) faster than standard k-means.

Using following command will cluster the features inside HDF5 file to generate a codebook. The clustered features will store inside pickle file.
```
python cluster_features.py --features_db output/features.hdf5 --codebook output/vocab.cpickle --clusters 1536 --percentage 0.25
```

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/clustered_features.png" width="200">

Figure 3: `vocab.cpickle` file ("codebook" or "vocabulary") contains 1536 cluster centers.

### Visualize features
The `visualize_centers.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/visualize_centers.py)) can help us to visualize the cluster centers from the codebook.

Using following command will create a visualization on each codeword inside codebooks (each centroid of clustered features).

```
python visualize_centers.py --dataset ukbench --features_db output/features.hdf5 --codebook output/vocab.cpickle --output output/vw_vis
```

This process takes about 60 - 90 mins to finish depend on the computers.

Here are a few samples (grayscale) of visualizing the features.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample1.jpg" width="300"> <img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample2.jpg" width="300">

Figure 4: Book-title features (left), Leaves-of-tree features (right).

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample3.jpg" width="300"> <img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample4.jpg" width="300">

Figure 5: Detailed grass features (left),  Car-light features (right).

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample5.jpg" width="300"> <img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/vis_sample6.jpg" width="300">

Figure 6: Store-logo features (left), car-dashboard features (right).

### Vector quantization
**This is the last step in building the bag of visual word (BOVW).**

There are multiple feature vectors per image by detecting keypoints and describing the image region surround each of these keypoints. These feature vectors are (more or less) unsuitable to directly applying CBIR or image classification algorithms.

What I need is a method to take these sets of feature vectors and combine them in a way that:
1. results in a single feature vector per image.
2. does not reduce the discriminative power of local features.

I use vector quantization to utilize visual vocabulary (or named codebook) to reduce this multi-feature representation down to a single feature vector that is more suitable for CBIR and image classification.

The `bagofvisualwords.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/information_retrieval/bagofvisualwords.py)) in `information_retrieval/` directory (inside `image_search_engine/image_search_pipeline/` directory) contains a `BagOfVisualWords` class that helps to extract BOVW histogram representations from each image in the dataset.

The `bovwindexer.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/indexer/bovwindexer.py)) in `indexer/` directory contains a `BOVWIndexer` class to efficiently store BOVW histograms in an `HDF5` dataset.

The `extract_bovw.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/extract_bovw.py)) is a driver file to handle looping over each of the images in `features.hdf5` dataset, constructing a BOVW for the feature vectors associated with each image, and then adding them to the `BOVWIndexer`.

Using following command will create a BOVW representation for each image by quantizing the associated features into histogram. Comparing to `features.hdf5` file in [previous](#extract-keypoints-and-descriptors) part, `bovw.hdf5` has much smaller size which only has 12.4 MB, as the figure shown below.  

```
python extract_bovw.py --features_db output/features.hdf5 --codebook output/vocab.cpickle --bovw_db output/bovw.hdf5 --idf output/idf.cpickle
```

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/bovw_database.png" width="200">

Figure 7: `bovw.hdf5` file contains bag of visual word representation for each image.

### Inverted indexing
The goal of this CBIR module is to build an efficient, scalable image search engine. In order to build such a scalable system, `redis` is used, which is a super fast, scalable in-memory key-value data structure store that I use to build an inverted index for the bag of visual words model.

`redis` can provide:
1. Built-in persistence: Either by snapshotting the database and writing it to disk or via log journaling. This implies that you can use `redis` as a real database instead of a volatile cache (like Memcached) — the data in your store will not disappear when you shutdown your server.
2. More datatypes: We can actually store data structures, including strings, hashes, lists, sets, sorted sets, and more. It quickly becomes obvious that `redis` is a very powerful data store, capable of much more than simple key-value storage.

Inverted indexing is borrowed from the field of Information Retrieval (i.e, text search engines). The inverted index stores mappings of unique word IDs to the document IDs that the words occur in.

In the context of CBIR, this allows us to:
1. Describe our query image in terms of a BOVW histogram
2. Query our inverted index to find images that contain the same visual word as our query.
3. Only compare images in our database that contain a significant number of visual words as the query.

Using redis not only ensure that we don’t have to perform an exhaustive linear search over all images in our dataset, it also speeds up the querying process, allowing our CBIR system to scale to millions of images rather than limit it to only a few thousand.

The redis database is using only about 1.9 MB of RAM to store 1000 images by checking the memory process.

The `redisqueue.py` ([check here](https://gurus.pyimagesearch.com/topic/building-an-inverted-index/)) inside `database/` directory (inside `image_search_engine/image_search_pipeline/` directory) will be used to interface with `redis` data store.

The `build_redis_index.py` will take `bovw.hdf5` dataset containing BOVW representations for each image in our dataset and build the inverted index inside `redis`.

Using following command while making sure that redis server is on will build a corresponding inverted index.

```
python build_redis_index.py --bovw_db output/bovw.hdf5
```

### Search performance
The `searchresult.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/information_retrieval/searchresult.py)) inside `information_retrieval/` directory defines a named tuple `SearchResult`, an object used to store:
1. Our search results, or simply, the list of images that are similar to our query image.
2. Any additional meta-data associated with the search, such as the time it took to perform the search.

The `dist.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/information_retrieval/dist.py)) inside `information_retrieval/` directory defines a `chi2_distance` function to compute distance between two histograms.

The `searcher.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/image_search_pipeline/information_retrieval/searcher.py)) inside `information_retrieval/` directory defines a `Searcher` class that can:
1. Use the inverted index to filter images that will have their chi-square distance explicitly computed by only grabbing image indexes that share a significant number of visual words with the query image.
2. Compare BOVW histograms from the short list of image indexes using the chi-sqaure distance.

The `search.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/search.py)) is the driver script that is used to perform an actual search. It will be responsible for:
1. Loading a query image.
2. Detecting keypoints, extracting local invariant descriptors, and constructing a BOVW histogram.
3. Querying the `Searcher` using the BOVW histogram.
4. Displaying the output results to our screen.

Using following command will start the search engine to return 20 closest images to the query image you choose.

```
python search.py --dataset ukbench --features_db output/features.hdf5 --bovw_db output/bovw.hdf5 --codebook output/vocab.cpickle --relevant ukbench/relevant.json --query ukbench/ukbench00364.jpg
```

In the UKBench dataset, since every subject has 4 relevant images with different viewpoints, the best performance will have top 4 images which are relevant to the query image and the worst will have none relevant to the query image. The top 20 results from search engine will be displayed. The displayed images has been resized for displaying purposes only.

Here are two samples that top 4 results are all relevant to the query image.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_sample1.png" width="800">

Figure 8: Query image ID: 364, search took: 0.91s.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_sample2.png" width="800">

Figure 9: Query image ID: 697, search took: 1.07s.

Here is a sample that top 3 results are relevant to the query image.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_sample3.png" width="800">

Figure 10: Query image ID: 819, search took: 1.67s.

Here is a sample that top 2 results are relevant to the query image.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_sample4.png" width="800">

Figure 11: Query image ID: 788, search took: 1.75s.

Here is a sample that only 1 results are relevant to the query image.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_sample5.png" width="800">

Figure 12: Query image ID: 333, search took: 0.64s.

### Evaluation
A point-based metric is implemented for evaluation. The system will get 1 point for each correct result in top-4 results. For example:
  * The system will receive 4 points if all four relevant images are in the top-4 results.
  * 3 points for three relevant images are in the top-4 results.
  * 2 points for two relevant images are in the top-4 results.
  * 1 point for only one relevant image in the top-4 results.
  * And 0 point for no relevant images in the top-4 results.

This scoring scheme is referenced from Nistér and Stewénius in their 2006 paper, *Scalable recognition with a vocabulary tree* ([reference](http://www-inst.eecs.berkeley.edu/~cs294-6/fa06/papers/nister_stewenius_cvpr2006.pdf)). According to their paper, a score ≥ 3 should be considered to be good, implying that on average, across all images in our dataset, we should be able to find at least three relevant images in the top-4 results.

The `evaluate.py` ([check here](https://github.com/meng1994412/CBIR/blob/master/image_search_engine/evaluate.py)) provide the evaluating process.

The figure below shows the evaluation results. The u stands for average and o stands for standard deviation.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_evaluation.png" width="600">

Figure 13: Evaluation results.

### Improvement
#### tf-idf
Term frequency-inverse document frequency, or simply tf-idf for short, is a numerical statistic borrowed from the field of Information Retrieval (i.e., text search engines), used to reflect how important a word or document is in a collection/corpus.

The tf-idf statistic increases proportionally with the number of times a visual word appears in an image, but is offset by the number of images that contain the visual word.

Therefore, a visual word that appears in many images in a dataset is much less discriminative and informative than a visual word that appears in only a few images in the dataset.

I implement tf-idf weighting on the bag of visual words representations, ultimately allowing to increase the accuracy of the CBIR system.

After implementing term frequency and inverted document frequency, the system is re-evaluated and Figure 11 shows the evaluation results.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/results/performance_evaluation_idf.png" width="600">

Figure 11: Evaluation results with tf-idf.

The system accuracy improve about 3.3%.

## Future Exploration
### Spatial Verification
Other than tf-idf, I also want to implement spatial verification with RANSAC algorithms to increase the accuracy of the CBIR system in the future.
