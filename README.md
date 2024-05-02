# [Speech Understanding] Major Project

This readme file contains detailed explanation of the code provided and instructions on how to run each file for successful reproduction of the reported results.

#### Environment requirements
First, note that in order for the provided code to run successfully, an environment with all the required packages must be installed. Consequently, we have provided a *'package_requirements.txt'* file which contains the name and respective version of all the packages used during this major project and are necessary to reproduce the results.

Second, note that the provided code is computationally and storage-wise extensive. This implies that it requires strong GPU-resources and large storage (for data, models etc.). Consequently, please run the provided code files on such a system only.

#### Pre-requisites
Now, note that there are data-wise pre-requisites of the code. This is listed as follows:

##### Data-wise pre-requisites
Now, the required data must be downloaded, unzipped and located in the specified path (data).
 - 'UrbanSound8K' dataset (data/ESC-50-master).
 - 'ESC50' dataset (data/UrbanSound8K).

### Instructions

Now, note the following set of instructions for the successful reproduction of provided results.

- Run the main.py file with the appropriate arguments for training. An example command for easier understanding is given as follows:

        ``python main.py --dataset esc50 --model_name astmodel_attentionsparse --epochs 10 --batch_size 4 --lr 1e-4``

- Run the eval.py file with the appropriate arguments for evalution. An example command for easier understanding is given as follows:

        ``python eval.py --dataset esc50 --model_name astmodel_attentionsparse --pretrained_path <path-to-pretrained-model>``

NOTE: Please contact the author in case of any discrepancy.