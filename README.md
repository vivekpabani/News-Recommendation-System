# News-Recommendation-System

The directory contains three subdirectories :
————————————————————————————————————————————

1. source_code : 8 pythons scripts , one each for each algorithm(kmeans, knn, nb, rank_classifier) + document.py script to represent the articles as a document object + a script to prepare tfidf and tfidfie from doc tf + util script + main.py script to run the application.
2. report : A PDF file documenting all the experiments details.
3. dataset : BBC dataset, divided into 5 subdirectories, one each for a topic, consisting 2225 documents in total.

To RUN:
———————

Change the working directory to source_code.

on command line : python main.py

Python version 2.7 is recommended.


Guide:
———————

The application will first train and test three classifiers on the dataset. It will report confusion matrix and statistics.

Once classifiers are ready, it will start recommendation interface. 

User will be given with ’n’ article titles to choose from. Based on the choice, ‘m’ recommended article titles will be displayed. ‘m’ and ’n’ can be chosen by user.(By default 5, in case of invalid input.) 

User can read the text of the recommended article by selecting the article number. 

User can read the original article by selecting option ‘o’ after recommendations are displayed. 

User can go back to the choice list by selecting ‘b’. 

User can refresh the original article list by selecting ‘r’. 

Select ‘q’ to quit the application.
