# DREAMER: Data REAdiness for MachinE learning Research

This work is published in _BMC Medical Informatics and Decision Making_ (https://doi.org/10.1186/s12911-024-02544-w).

**DREAMER** is a computational framework to assess readiness level of datasets for machine learning. It uses Python Pooling Multi-Processing method to maximize the benefit from existing computational resources. One can determine the number of CPU cores in the DREAMER json configuration to run a parallel job.

## Install DREAMER
To install the DREAMER use the follolwing command: 

`pip install git+https://github.com/vkola-lab/DREAMER.git`    
     
The dependency packages for running DREAMER are:    
        ***- pandas (1.5)     
	- numpy (1.23)      
	- scikit-learn (1.2)    
	- scipy (1.10)      
	- numba (0.56)***     


## How to run DREAMER?
The input dataset of DREAMER should be numerical in CSV tabular format and consists of one class (label) column.
These are steps for running DREAMER on your master dataset:
  
1. Put the CSV dataset within your root folder. 
2. Setup json file parameters and put it within the root folder. The json file name must be **"DREAMER_Config.json"**.
3. Run DREAMER process using your own main file by calling **dreamer.run()** function after creating DREAMER object "dreamer"
with these three simple lines code (see "DREAMER_Run.py" file in the tests folder):      
`from DREAMER import DREAMER`    
`dreamer = DREAMER()`      
`dreamer.run()`             
4. The results inculding cleansed version of the master dataset and some statistical reports will be generated in the output
folder.

## DREAMER json configuration
The DREAMER_Config.json file has three main features: **"search_space", "data_settings", and "free_params"**. Please see the
sample DREAMER configuration file to setup parameters according to your own dataset. In the following, the specification
of each json file features will be described:

### 1. search_space
This feature has two parameters: **"subtables" and "cores"**. Parameter "subtable" specified the number of
random subtables that are generated during random exploration of the DREAMER process. It could be between 1000 to 1,000,000.
The values more than 1,000,000 could be very time-consuming depending on the size of master dataset. Parameter "cores" indicates
the number CPU cores that are used by DREAMER to perform multiprocessing task for running. If you set this parameter to zero, 
DREAMER will use all CPU cores during random exploration and all cores will be in 100% usage. The running time of the whole process
depends on this two parameters and also the size of master dataset (number of rows and columns).

### 2. data_setting
This feature has three parameters: **"file_name", "target_column", "output_folder"**. Parameter "file_name" is 
the name of master dataset file which should be a csv file such as "MyDataset.csv". Parameter "target_column" specifies the name 
of class (label) column. DREAMER uses this column for supervised assessment of the dataset. And parameter "output_folder" indicates 
the name of folder that will be created in the root folder to store all results including cleansed data and reports.

### 3. free_params 
This feature has five parameters: **"rows_exclusion_ratio", "columns_exclusion_ratio", "clustering_weight",
"classification_weight", and "precision"**. "rows_exclusion_ratio" and "columns_exclusion_ratio" indicates the maximum ratio 
of rows and columns that could be deleted from the master dataset to obtain the cleansed dataset and should be a number between 0 and 1.
"clustering_weight" and "classification_weight" indicates the weight of classification and clustering tasks to calculate the overal precision
of the dataset and should be a number between 0 and 1. If we set both parameters as 0.5, it means that the weights of both supervised and
unsupervised methods for calculating the readiness level of the dataset are the same. Finally, the parameter "precision" shows the precision
value of floating numbers to show the statistical values in the reports. The value 4 for this parameter could be reasonable.

## DREAMER outputs
In the output folder you will see five different output files which shows the cleansed version of the master dataset and some statistical
measures of both master and cleansed datasets for comparison. You can also find a file namely **"Time.txt"** that shows the total running time
of the DREAMER process in minutes. The file **"CleanData.csv"** is the final cleansed version of the original dataset which includes class column
and some of the rows and columns from master dataset. The indexes of selected rows are demonstrated in the first column of this file.

# Web development framework
We have developed **DREAMER**, a web-based tool accessible at https://dreamer.bu.edu, which facilitates users in registering on our platform, uploading their datasets, assessing their Machine Learning (ML) readiness, and obtaining cleansed datasets as outputs. The intuitive user interface accommodates the uploading of a master CSV dataset. Upon initiation, this action initiates an Application Programming Interface (API) connection, generating a JSON configuration file containing DREAMER parameters pertinent to the master dataset, subsequently transmitting it to our server. The backend system then assumes control, executing the core DREAMER processes on the dataset, resulting in a sanitized CSV file accompanied by comprehensive reports and statistical analyses. Upon conclusion of the DREAMER procedures, users are promptly notified via email and granted access to download the complete package, inclusive of the cleansed dataset, reports, and data readiness metrics. The overal architecture of the DREAMER web framework is presented below:

<img src = "https://github.com/vkola-lab/DREAMER/blob/main/Web_framework.svg">


