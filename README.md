## Machine Learning [EECS 5750]

## Case Study

## on

## Vessel Prediction from Automatic

## Identification System (AIS) Data

### Submitted by:

### Neeraj Shrestha

### R

### Prashish Paudel

### R

### Neeraj.Shrestha@rockets.utoledo.edu

### prashish.paudel@rockets.utoledo.edu

### Date: 12/15/


### Vessel Prediction from Automatic Identification System

### (AIS) Data

### Introduction

Automatic Identification System (AIS) is an automated, self-contained tracking system widely
used by Vessel traffic services in the maritime system to exchange navigational data. In this case
study, we were given Automatic Identification Systems (AIS) data generated by maritime vessels.
Marine mobile service identity (MMSI) number is assigned to each vessel which uniquely
identifies it and helps in tracking it over time. The goal of this case study is to track the movements
of various vessels based on location reports without the use of MMSIs. To achieve this objective,
we applied various preprocessing techniques to the features and tried many algorithms which we
will discuss throughout this report.

### Data Preprocessing

For this case study, we were provided with three datasets namely set1.csv, set2.csv, and
set3noVID.csv. Each row of all the datasets corresponds to an observation of a single maritime
vessel at a single point in time. Each dataset had five features (SEQUENCE_DTTM, LAT, LON,
SPEED_OVER_GROUND, COURSE_OVER_GROUND). SEQUENCE_DTTM is the time of
reporting in the form of hh:mm:ss in Coordinated Universal Time(UTC). LAT and LON are the
latitude and longitude of vessel position in decimal degrees. SPEED_OVER_GROUND is the
vessel speed in tenths of a knot (nautical mile per hour). COURSE_OVER_GROUND is the angle
of vessel movement in tenths of a degree. Datasets set1.csv and set2.csv have VID label which
was provided to us for testing purposes whereas dataset set3noVID.csv has all VID set to zero and
was kept for our evaluation.

We started our case study by studying datasets. We found that the value of SEQUENCE_DTTM
lies between 14:00:00 and 17:59:58. The value of latitude is around 36 and longitude is around –

76. We found that there is no pattern in SPEED_OVER_GROUND and
COURSE_OVER_GROUND and a particular vessel can have any value for those at different
SEQUENCE_DTTM.

Our preprocessing is based on the assumption that vessels with the same VID moving at a certain
speed at a certain angle end up in the same location at a certain time. And it can be seen that our
assumption is not always true because we can see in the data that vessels with the same VID have
different SPEED_OVER_GROUND and COURSE_OVER_GROUND values at different times.
But we thought that this is the best approach that we could follow and continued with it. The idea
of this approach was to find the final latitude and longitude of the vessel using the current latitude,
longitude, speed, angle, and time of the vessel and use the find latitude and longitude only as the
feature to predict the clusters. To implement this, we first created a function def
getNewCoordinates(lat,lon,time,speed,angle) where we first converted the angle into radians. We


then computed the distance in km the vessel has traveled using speed and time difference between
the current time and end time in seconds (64796). Then, we computed the final latitude and
longitude using the radius of the earth, computed distance, and implemented those values in a
formula that is similar to the Haversine formula. Then we created a new feature matrix using only
the values of final latitude and longitude.

We also tried standardization techniques like StandardScaler but it was not suitable for our features
and it decreased the Adjusted Rand index(ARI). We also tried to use unsupervised feature
extraction techniques like Kernel Principal Component Analysis, which is used to project the
features into higher dimensional feature space, but it did not bring any improvement to the result.

### Algorithm Selection

We implemented and tested many unsupervised models. We only used unsupervised models
because the VID (label) in three datasets is not consistent, and each VID does not represent the
same vessel in different datasets. We tried different clustering algorithms like K-means, Spectral
clustering, Hierarchical clustering, DBSCAN, and Gaussian Mixture Model.

The first algorithm that we tried was the K-means algorithm. For the first dataset, we used k=
and got an Adjusted Rand Index of 0.15068961341852738 and for the second dataset, we used
k=8 and got an Adjusted Rand Index of 0.6156239573031684. For the first dataset, our ARI was
less than the ARI of the baseline algorithm given to us whereas it was higher for the second dataset.
So, the performance of K-means was unpredictable.

The second algorithm that we tried was spherical clustering which was too slow and it did not give
us any result even running for few minutes. So, we dropped the idea of using spherical clustering.

The third algorithm that we tried was Agglomerative Clustering. For the first dataset, we used the
number of the cluster as 20 and got an Adjusted Rand Index of 0.16336702386223215 and for the
second dataset, we used k=8 and got Adjusted Rand Index of 0.6169653937867234. For the first
dataset, our ARI was less than the ARI of the baseline algorithm given to us whereas it was higher
for the second dataset. So, the performance of Agglomerative Clustering was inconsistent.

We read many research papers and online articles about the use of unsupervised machine learning
models in vessel prediction and found most of them use DBSCAN. But when we tried using
DBSCAN, we got an ARI value of 0 in both datasets.

The last model that we tried was the Gaussian Mixture Model with the expectation-
maximization(EM) algorithm. For the first dataset, we used the number of a cluster as 20 and got
an Adjusted Rand Index of 0.1621999721138152 and for the second dataset, we used k=8 and got
an Adjusted Rand Index of 0.6291664420304007. For the first dataset, our ARI was less than the
ARI of the baseline algorithm given to us whereas it was higher for the second dataset. So, the
ARI of the Gaussian Mixture Model was highest among all the algorithms that we used.


### Selected Algorithm

We selected the Gaussian mixture model with expectation-maximization (EM) algorithm for
fitting a mixture of Gaussian models. We are using an unsupervised algorithm that does not use
any training data. When we studied the paths of vessels after our preprocessing, we found them to
be somewhat spherical. That is why we used the Gaussian mixture model with spherical covariance
type which is suitable for spherical clusters. It is a flexible and scalable model which results in
different results for different parameter values. After manually changing the parameters and testing
the ARI, we got the maximum ARI with the following parameters.

```
 n_components: It is the number of mixture components. Here it is given by the number of
vessels (20 for the first dataset and 8 for the second dataset). It can also be chosen
manually using different analysis.
 covariance_type: Out of full, tied, diag and spherical, we used spherical because it was
suitable for our preprocessed features
 init_params: Out of kmeans and random, we chose kmeans so that responsibilities can be
initialized using it.
 random_state: It controls the random seed. We set it to 100.
```
We performed silhouette analysis for selecting the number of clusters in our dataset. Silhouette
analysis is also helpful in the study of the separation distance between the resulting clusters. While
performing silhouette analysis for various no of clusters, we found 10 as the optimal no of clusters
with a silhouette coefficient of 0.38. Although for cluster no =13, we obtained the highest
silhouette coefficient of 0.40 but various groups of the clusters had a negative value. Since for


cluster no =10, fewer groups had negative values and the silhouette coefficient was also
comparable to the highest silhouette coefficient (cluster =13), we chose our cluster number = 10.

### Evaluation

In evaluating the effectiveness of our algorithm as a multiple target tracking system, the goal is to
measure the distance between two sets of tracks i.e the set of ground truth tracks and the set of
estimated tracks, produced by our algorithm and use it to cluster the vessels. We have used the
Adjusted Rand Index (ARI) and silhouette analysis to evaluate the efficacy of our clustering
technique. These two evaluation techniques do have their limitations, and we could use more
robust evaluation mechanisms that corroborate well with our case study. Prevailing weather
condition is a significant factor in the behavior of a vessel. Inclusion of which should be done to
the existing model to increase the effectiveness of tracking the trajectory of the vessels. Since, a
method here is data-driven, having more amount of data will help in increasing the accuracy of the
model. With the help of more elaborate data metrics such as headings, weather, dimension of
vessels, vessel’s start, and endpoint, we could enhance the quality of the clustering as these metrics
are very important for real-world implementations. As we know, the quality of the training data
can be ensured by a pre-processing procedure to remove the anomaly in the AIS trajectory Data.
The proposed method is indeed in probable need of a more robust methodology for thorough
determinations despite having the ability to weed out the outliers from the trajectory. Beyond, its
use in maritime vessel tracking, our algorithm can also be useful for modeling data from other
moving objects such as data from vehicles, data from the UAV's, data from pedestrians.

The main implication of our algorithm is the assumption on which it is based. Our preprocessing
is based on the assumption that vessels with the same VID moving at a certain speed at a certain
angle end up in the same location at a certain time. And this is not true in most cases because the
vessels change their speed and direction during their travel. They can also become stationary after
some time. This can also be verified from our data as we can see in the dataset that vessels with
the same VID have different SPEED_OVER_GROUND and COURSE_OVER_GROUND values
at different times. Our algorithm best works with the dataset where vessels do not change their
speed and direction during a certain interval of time. But this is not the case most of the time. So,
opposite to our assumptions, the vessels with the same VID will end up in different locations and
our algorithm will treat the same vessels as different ones and cluster them differently.

Practical issues like gaps in data, e.g., no data available from all vessels for 10 minutes would not
affect much to our algorithm because we are assuming that the speed and angle of vessels do not
change over time. If we need those missing data, we can generate those data using the interpolation
technique. Though the generated data may not be as accurate as of the actual data, it might be
helpful to improve the performance of the algorithm. Another important issue with AIS data is its
inaccuracy. We can handle this issue by using anomaly or outlier detection algorithms like
isolation forest which isolates the outlier from the given features by randomly picking them and
then randomly selecting a split value between the max and min values of that feature.


### Conclusion

This is how the model for tracking motion of different vessels without Vessel ID was developed
in this case study. Our preprocessing technique was carried out for it to be done. Tested candidates
for unsupervised clustering algorithms were K-means, Spectral Clustering, DBSCAN, and
Gaussian Mixture. Though more features would have helped to improve the accuracy of our
algorithm nonetheless with the given feature the best outcome was achieved by the Gaussian
Mixture model with Expectation-Maximization algorithm for tracking the position of different
vessels. The accuracy of our algorithm highly depends upon the trajectory data of the vessels.
Based on the assumption used for preprocessing, our algorithm can correctly cluster the vessels
which have straight line trajectory.


