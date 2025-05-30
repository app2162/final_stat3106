# Clustering U.S. counties on Damage to Buildings Expected Because of Natural Disasters
This project seeks to cluster counties based on their Expected Annual Loss Rate (EAL) for Buildings (in 2022 dollars) for 17 different natural disaster hazards (coastal flooding, avalanche, earthquake, etc). The goal is to find outlier counties in each cluster and determine if they are exeptionally at-risk or particularly underresourced compared to other counties in their cluster. This is important because counties that are relatively similar in terms of hazard risk but are outliers within their clusters can be unlike their neighbors and get overlooked in regional planning, so it is important to find gaps and places that could fall through the cracks. 

## Structure of Repo
- ingest_data includes ingestion of data, data checks for missing values, NaNs, etc, log transformation of EAL, creating housing fragility variables, merging demographic/FEMA/geographic data
- features_algo includes normalization by column, attempted PCA, K-means, DBSCAN clustering, clusters, risk profiles, assessing outliers by cluster
- validation includes comparing clustering to dumb but reasonable approach, understanding clusters and outliers to clusters

## Data
The first data set comes from the [FEMA National Risk Index](https://hazards.fema.gov/nri/data-resources), which defines risk as the potential for negative impacts as a result of a natural hazard (earthquake, flood, winter weather, hail, etc), and includes the feature of interest--"Expected Annual Loss Rate - Building"--- which is designed to reflect the average expected annual percentage loss for the building value within each county, controlled for community size. The second data set comes from the U.S. Census American Community Survey 2021 5-Year Estimates, which provides demographic data for each county in the US and was obtained from socialexplorer.com. 

