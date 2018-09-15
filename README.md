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
⋅⋅⋅Fast Hessian keypoint detector algorithms
⋅⋅⋅Local scale-invariant feature descriptors (RootSIFT)
* Feature storage and indexing
⋅⋅⋅Structure HDF5 dataset
* Clustering features
⋅⋅⋅K-means algorithms

## Results
After storing the keypoint detectors and local invariant descriptors of each image in HDF5. We will have a HDF5 file shown below.
