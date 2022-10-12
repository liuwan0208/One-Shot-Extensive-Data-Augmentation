## 1 Paper
Liu, W., Lu, Q., Zhuo, Z., Liu, Y., Ye, C., 2022. One-shot segmentation of novel white matter tracts via extensive data augmentation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 133–142.


## 2 Prerequisites
### 2.1 Environment and Software
* Linux & OSX, Python>=3.6
* [Pytorch](https://pytorch.org/)
* [Mrtrix 3](https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html) (>=3.0 RC3)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) 
### 2.2 Install Baseline Code and BatchGenerators
* We use [TractSeg](https://github.com/MIC-DKFZ/TractSeg/) as the baseline, and install it from local source code:
```
git clone https://github.com/MIC-DKFZ/TractSeg.git
pip install -e TractSeg
```
* Install `BatchGenerators`:
```
git clone https://github.com/MIC-DKFZ/batchgenerators.git
pip intall -e batchgenerators
```
* Create a file `~/.tractseg/config.txt`, and write the path of your own directory in config.txt, e.g. `working_dir=Your_OutputPath`.


## 3 Run Our Code
### 3.1 Download Code
* Download our code as zip, and unzip it.
* Save our code in the same directory of TractSeg code, i.e. `Your_CodePath`.
### 3.2 Data Preparation
* Download the [HCP scans](https://db.humanconnectome.org) and the [gold standard of WM tracts](https://db.humanconnectome.org).
* Extract the input peaks images from dMRI scans with 'Your_CodePath/TractSeg_Fewshot_Pretrain/`bin/Generate_Peaks.py`'.
* Arrange the peaks and annotations of different subjects to the following structure:
```
Your_DataPath/HCP_for_training_COPY/subject_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks.nii.gz       (Reference bundle masks; shape: [x,y,z,nr_bundles])
Your_DataPath/HCP_for_training_COPY/subject_02/
      ...
```
* Use 'Your_CodePath/extract_tract_label.py' to extract the annotations of the existing WM tracts from annotation file bundle_masks.nii.gz and save it as bundle_masks_60.nii.gz；Use 'Your_CodePath/extract_tract_label.py' to extract the annotations of the 4 (6/12) novel WM tracts from bundle_masks.nii.gz and save it as bundle_masks_4.nii.gz (bundle_masks_6.nii.gz/bundle_masks_12.nii.gz). Then the peaks and annotations in `HCP_for_training_COPY` fold are arranged as:
```
Your_DataPath/HCP_for_training_COPY/subject_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks_60.nii.gz       (Reference bundle masks; shape: [x,y,z,60])
            '-> bundle_masks_4.niigz       (Reference bundle masks; shape: [x,y,z,4])
            '-> bundle_masks_6.nii.gz       (Reference bundle masks; shape: [x,y,z,6])
            '-> bundle_masks_12.nii.gz       (Reference bundle masks; shape: [x,y,z,12])
Your_DataPath/HCP_for_training_COPY/subject_02/
      ...
```
*  Select the subjects used for pretraining, training, and test.
* For the pretraining subjects, remove the non-brain area of the peaks image and bundle_masks_60.nii.gz in `HCP_for_training_COPY` fold with 'Your_CodePath/Remove_Nonbrain.py', and arrange the data to the following structure:
```
Your_DataPath_pretrain/HCP_preproc/subject_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks_60.nii.gz    (Reference bundle masks;  shape: [x,y,z,60])
Your_DataPath_pretrain/HCP_preproc/subject_02/
      ...
```
* For the single training subject, use 'Your_CodePath/Data_Augmentation' to generate synthetic annotated scans from the single scan in `HCP_for_training_COPY` fold; Remove the non-brain area of the peaks image and the novel tract annotation of the synthetic scans, and arrange the data to the following structure (use RC1 and 4 novel tracts as an example):
```
Your_DataPath_train/RC1_4/HCP_preproc/daug_01/
            '-> mrtrix_peaks.nii.gz       (mrtrix CSD peaks;  shape: [x,y,z,9])
            '-> bundle_masks_4.nii.gz    (Reference bundle masks;  shape: [x,y,z,4])
Your_DataPath_train/ RC1_4/HCP_preproc/daug_02/
      ...
```
### 3.3 Network Pretraining
* Adapt 'Your_CodePath/TractSeg_Fewshot_Pretrain/tractseg/libs/system_config.py' and modify `DATA_PATH` to 'Your_DataPath'.
* Adapt 'Your_CodePath/TractSeg_Fewshot_Pretrain/bin/ExpRunner' with the list of pretraining subject IDs.
* Set the temporary enviroment variable in terminal to our code path:
```
export PYTHONPATH=$PYTHONPATH:Your_CodePath/TractSeg_Fewshot_Pretrain
```
* `Train` the network:
```
python run Your_CodePath/TractSeg_Fewshot_Pretrain/bin/ExpRunner
```
* The `training output` is saved in 'Your_OutputPath/hcp_exp/my_custom_experiment'.
### 3.4 Network Training
* Modify the file 'Your_CodePath/TractSeg_Fewshot_Pretrain/tractseg/experiments/base.py', including the specific training stage (warmup stage or fine-tune stage), path of the pretrained model, the path of the trained data.
* Adapt 'Your_CodePath/ TractSeg_Fewshot_Warmup /bin/ExpRunner' with the list of traning subject IDs.
* Set the temporary enviroment variable in terminal to our code path:
```
export PYTHONPATH=$PYTHONPATH:Your_CodePath/TractSeg_Fewshot_Warmup
```
* `Train` the network:
```
python run Your_CodePath/TractSeg_Fewshot_Pretrain/bin/ExpRunner
```
### 3.5 Test the Trained Model
* Adapt 'Your_CodePath/TractSeg_Fewshot_Warmup/ bin/ExpRunner_test.py' with the trained model and the list of test subject IDs.
* Set the temporary enviroment variable in terminal to our code path:
```
export PYTHONPATH=$PYTHONPATH:Your_CodePath/TractSeg_Fewshot_Warmup
```
* `Test` the network:
```
python run Your_CodePath/TractSeg_Fewshot_Warmup/bin/ExpRunner_test.py
```
* Fuse the results from the model trained with different data augmentation strategy with ‘Your_CodePath/ seg_ensemble.py’
