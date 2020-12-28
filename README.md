# Text-Clustering-Evolution-Map
The model purposed K-means and Hierarchical that unsupervised learning based methodology to cluster the patent. Also, based on key phrases and population years relationships to visual evolution map.

# Website Server
http://140.114.53.218/python/programs/clustering/

Input date is related sample dateset
1. Put the patent in the csv file according to the input_data format.
Note: Please confirm that each patent is put in only one line (use notepad ++ open the csv file to confirm-replace \n).
Suggestions: In addition to the original Title, Abstract, and Claims of the patent,  you can also use DWPI content to increase accuracy.
2. Web will show 200 words that have the highest TF-IDF scores. 
3. You should pick up 50~100 words from top200.csv which was generate in previous step , and then change it name into Top.csv
4. Compress your all data into a zip file before uploading.
5. Output data clustering result-rank1~3.csv – the clustering result which have top 3 silhouette score 
