# Content-Based Image Retrieval Syetem
## Project Objectives
* Extract keypoint detectors and local invariant descriptors of each image in the dataset and store them in HDF5.
* Cluster the extracted features in HDF5 to form a codebook (resulting centroids of each clustered futures) and visualize each codeword (the centroid) inside the codebook.
* Construct a bag-of-visual-words (BOVW) representation for each image by quantizing the associated feature vectors into histogram using the codebook created.
* Accept a query image from the user, construct the BOVW representation for the query, and perform the actual search.

## Software/Package Used
* Python 3.5
* [OpenCV](https://docs.opencv.org/3.4.1/) 3.4
* [Imutils](https://github.com/jrosebr1/imutils)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [HDF5](https://www.h5py.org/)
* redis

## Algorithms & Methods Involved
* Keypoints and descriptors
  * Fast Hessian keypoint detector algorithms
  * Local scale-invariant feature descriptors (RootSIFT)
* Feature storage and indexing
  * Structure HDF5 dataset
* Clustering features to generate a codebook
  * K-means algorithms
* Visualizing codeword entries (centroids of clustered features)

## Results
### Extract keypoints and descriptors
Using following command will store the keypoint detectors and local invariant descriptors of each image in HDF5. We will have a HDF5 file shown below.
```
python index_features.py --dataset ukbench --features-db output/features.hdf5
```

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/hdf5_database.png" width="200">

The following picture shows the interior structure inside HDF5 file:

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/hdf_database_layout.png" width="500">

The `image_ids` dataset has shape (X,) where X is total number of examples in dataset (In this case, X = 1000). And `image_ids` is corresponding to the filename.

The `index` dataset has shape (X, 2) and stores two integers, indicating indexes into `features` dataset for image i.

The `features` dataset has shape (Y, 130), where Y is the total number of feature vectors extracted from X images in the dataset. First two columns are the (x, y)-coordinates of the keypoint associated with the feature vector. The other 128 columns are from RootSIFT feature vectors.

## Cluster features
Using following command will cluster the features inside HDF5 file to generate a codebook. The clustered features will store inside pickle file.
```
python cluster_features.py --features-db output/features.hdf5 --codebook output/vocab.cpickle --clusters 1536 --percentage 0.25
```

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/clustered_features.png" width="200">

## Visualize features
Using following command will create a visualization on each codeword inside codebooks (each centroid of clustered features).
```
python visualize_centers.py --dataset ~/Desktop/ukbench_sample --features-db output/features.hdf5 --codebook output/vocab.cpickle --output output/vw_vis
```
This process takes about 60 - 90 mins to finish depend on the computers.

Here are a few samples (grayscale) of visualizing the features.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/vis_sample1.jpg" width="400">

The sample 1 looks like a book-title feature.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/vis_sample2.jpg" width="400">

The sample 2 looks like a leaves-of-tree feature.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/vis_sample3.jpg" width="400">

The sample 3 looks like a detailed grass feature.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/vis_sample4.jpg" width="400">

The sample 4 looks like a car-light feature.

<img src="https://github.com/meng1994412/CBIR/blob/master/image_search_engine/output/vis_sample5.jpg" width="400">

The sample 5 looks like a store-logo feature.
