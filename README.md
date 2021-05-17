# FelixML_New

Instructions to make a VAE from the very start:

1. Organise the data:

Do instructions 1.1 to 1.3 on the linux desktop account, 1.3 onwards use the PC.

X2Go is installed on this PC.

1.1 Move raw data

All the raw data needs to be put into 1 folder called for example "AllRawData", this needs to contain the simulated files with labels 1, 2, 3 ect... (no worries if there is a missing number). This is where the 10000 or so files will go. 

For some reason there are 2 simulations which are not complete from the 2nd round of Felix simulations. Inside 2001-3000.tar.gz and 5001-6000.tar.gz, files 2017 and 5550 do not work. (2017 contains strange files and 5550 does not contain StructureFactors.txt). It is important that each file contains the correct files, if some are missing, it causes head aches. So with the new simulations, there needs to be a check if each folder contains all those files needed.

1.2 Extract the data

Open program "FelixML_New/preprocessing/lattice_creation/VAE_Preprocessing_000_AllInOne.ipynb".

This program extracts the data from the .bin (000 beam only), copys the .cif and StructureFactor.txt files into 3 folders called Data CifFolder and StructureFactor.

The first thing you need to do is make an empty folder called "VAE_000". Next you need to copy the path of "AllRawData" into Path = "" variable and copy the path of "VAE_000" into NewPath = "".

This may take some time to run.


1.3 Move data from linux desktop account to the PC

Make sure there is enough space on the SSD, do not put the data on the HDD. The SSD is located at Home.

Download the folders Data, CifFolder and StructureFactor onto the desktop using Filezilla. This may take several hours.

1.4 Fix file path names

Since files have been moved to a different locations, a file called FilePaths.txt inside the folder StructureFactor has to be modified. Inside FilePaths.txt has lists of paths to the files inside the folder Data, so you need to Ctrl + h (replace) the first part of the string with the new path on the PC.

For example:
the line 1st line "/homeLinuxAccount/Data/1/Input.txt /homeLinuxAccount/StructureFactors/1.txt"

would have to be changed to

"/homePC/Data/1/Input.txt /homePC/StructureFactors/1.txt"

where homeLinuxAccount and homePC are the paths to the folder which contains Data and StructureFactor on the linux desktop account and the PC.

1.5 Create the unit cells from the structurefactor.txt files

This program was quite computaionally expensive, so multithreading was used by using #pragma in C.

Open program "FelixML_New/preprocessing/lattice_creation/create_electron_dist.c"

This program reads from a file called FilePaths.txt inside the folder StructureFactor. So you need to copy the file path of FilePaths.txt into the variable called PathOfPath. FilePaths.txt contains a list of paths which the C program uses, inside the file you need to check the number of lines (or see how many files are inside the folder "Data"). You need to set the variable NumberImages = to that number.

run the program by using the 3 lines in a terminal located where the C program is:

export OMP_NUM_THREADS=16

gcc -o RunFile CreateElectronDist.c -fopenmp -lm

./RunFile

This program also takes some time

1.6 Change .txt outputs into .npy files

Everytime you want to use jupyter notebook on the PC, you need to run these 4 lines the terminal at the location "ug-ml@ugml--pc:~/tmp_git/FelixML_New"

source /home/ug-ml/felix-ML/env/bin/activate

echo 1 | sudo tee /proc/sys/vm/overcommit_memory

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

jupyter notebook



The C program from 1.5 outputs .txt files, this needs to be changed to .npy files.

Open program "FelixML_New/preprocessing/lattice_creation/txt_to_npy.ipynb"

Change Path = "" into the path of the folder called "Data"

This program is very fast

1.7 Normalise the unit cells

The VAE works well when inputs/outputs are of order 1. But the unit cell contain values which usually vary between +/- 200. To make things more complicated, there are several files (of order 10) which contain values which are of order 10000. Jeremy and I honestly have no clue why, so we just removed them.

These "outliers" are not deleted, but moved into a new folder called "Outlier" which you can decide the location of (Make the folder Outlier anywhere).

Open program "tmp_git/FelixML_New/preprocessing/lattice_creation/Normalise_Unitcells.ipynb"

The program identifies these outliers by checking if the maximum or minimum values in the unit cell are some number of standard deviations away from the average.

Before you run this program, I recommend you make a backup copy of the folder "Data" just incase.

Step 1: change Path = "" into the path containing "Data" then Run that cell
Step 2: SD is the number of standard deviations, 5 seemed to be a good value, change OutlierPath = "" to the path where your Outlier folder is. Run that cell.
Step 3: Run the cell which does Normalise()
Step 4: Do not run the last cell, this is for multiple beams. You are now done with this program.

All your unit cells and LACBED pattern images are now normalised between [0, 1] and [0, 2] respectivly.

1.8 Create dataset

Open Program "tmp_git/FelixML_New/preprocessing/lattice_creation/RemoveSimilarImages.ipynb"

Step 1: Change Path = "" to path conataining folder "Data", run first, second, thrid and fourth cell. This may take some time, you can see the progress by looking at the output of the cell, the output needs to go to the number of crystals.

Step 2: This cell will identify which LACBED patterns look similar, if two crystals have a log loss defined as-
sum_ij[(log(1+Image1_ij) - log(1+Image2_ij))^2] less than a value of MS_crit, then the crystal with the larger index is removed.

We chose a MS_crit of 0.1, however, more work could be done to justify this value. Run this cell

Step 3: Run the next cell which prints 2 things.

Step 4: The training, validation and test datasets are created by randomly selecting the remaining crystals in ratios of [0.85, 0.1, 0.05]. 6 files will be created, validation, training, testing each having a file with input and output (3 x 2 = 6). Each of these files contains file paths inside the folder Data. The VAE will read these 6 files so it can read the data at those locations.

To makes these files, run the 2 cells with ShuffleIndex and the next one after that.

Step 5: To save these files, create a new folder which is called "FilePaths" anywhere and copy the path location to NewPath = "". Now set Name = "0point1", because MS_crit was set to 0.1 (feel free to name it something different though). Run that cell.

Step 6: Ignore the future cells, this is an outdated most similar program.




2. The VAE

Open Program "tmp_git/FelixML_New/models/generation/direction_000/generator_zmcc.ipynb"

Step 1: Run cells 1-7. These cells define functions used in the training processess, as well as the VAE class and its component classes.

Step 2: In cell 8, there are several variables you will need to assign paths to. These are:
	data_path : This is the folder "FilePaths" made in section 1.8 Step 5. This folder contains text files that list paths to the training, validation and test data.
	save_path : This is the directory in which to save the best model (the model with the lowest validation loss) during training.
	best_model_name : This is the name that will be given to the best model when it is saved.

Step 3: The variables TrainingPathsInput, TrainingPathsOutput, ValidationPathsInput, ValidationPathsOutput, TestPathsInput, and TestPathsOutput all require a string as an argument to their assigning function (gen_paths_fromfile) which is the path to the files created in section 1.8 step 5. You will need to change the strings appropriately if you decided to go with a naming convention different to "0point1". Otherwise, you can leave these as they are.

Step 4: Assign the desired batch_size, epochs and patience values. 

Step 5: Run this cell. The VAE should begin training until it reaches its patience or epoch limit. At the end of each epoch, if the current model has a lower validation loss than the best model, the current model will be saved as the best model. This model can then be loaded into an auxillary program for further analysis while training continues (if desired).


3. Analysis




