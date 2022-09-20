Please follow the instructions to run the code



To process test file and to run the webapp for the predictions;

prerequisite : python 3.9

1) Extract the zip contents and locate the run.sh file.

2) Run the below command with absolute path of testing file name like below(please make sure test file name contains testing);
            sh run.sh /tmp/2016-09-19_79351_testing.csv

Once you run the command it will compare the actual values and predictions and metrics will be printed 
and webapp will be started. Navigate to browser the type IP and port.











Docker:
If python is not installed you can use build the docker file and run the below commands;

Run this command where the dockerfile is located
1) docker build -t dstask_vjagannath -f ./Dockerfile ./

2) docker run -p 8050:8050 -v [host directory]:/tmp -ti dstask_vjagannath /bin/bash

3) cd /tmp/[host directory]

4) sh run.sh test_file_name

















########### To rerun for next coming months follow the below steps###########
1) run datasets.py file while giving the month as option to prepare the test set to predict
               python datasets.py future_month

2) run the training file
               python train.py

3)             python predict.py

4)             python compare.py

5) To see the results for each user 
               python app.py