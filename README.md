# DL-SpiQA
Quality assurance tool for dosimetric target verification of spine radiation therapy treatment plans.

System requirements:
PC with Linux-based operating system and dedicated graphics processing unit for auto-segmentation. Must have TotalSegmentator installed: https://github.com/wasserth/TotalSegmentator

System recommendations:
Ubuntu 20.04 or later with a 16GB GPU or better and at least 16GB RAM.

Usage:
Users must first edit the directory paths declared in lines 786-789 of the source code for DL-SpiQA.py.

Line 786: dataPullPath points to a data input folder containing deidentified patient data in NRRD format.
Line 787: dataPushPath points to a destination folder to store results (by default, this is the same as the input folder).
Line 788: workingRoot specifies an empty folder on the local machine to serve as a working directory for file handling.
Line 789: aiModelPath points to the location of the TotalSegmentator model.

The data input folder (labeled as "spine_data" in the below example) must have the following tree structure:

spine_data
├─ Patient1
│  ├─ CT
│  ├─ DOSE
│  ├─ RT Plan Data.txt
├─ Patient2
├─ Patient3
├─ Patient4
├─ vert_volume_statistics.csv


The DL-SpiQA workflow will process any arbitrary number of spine RT data within the same batch. If a single patient has two or more associated CTs, they must be separated into two different folders with different patient names.
The text file for each patient, "RT Plan Data.txt", must include the RT plan label (or any indication of the targeted vertebral levels) as well as the prescription dose.
Example contents of an arbitrary "RT Plan Data.txt" file are shown below:

RT Plan Label: A1_T6-T8
Prescription Dose (Gy): 35.0

The labels "RT Plan Label: " and "Prescription Dose (Gy): " must remain unchanged.

The vert_volume_statistics.csv file contains a statistical summary of a vertebral volumes from a large number of patients. The file must be contained inside the data input folder along with the patient data folders.


Once the setup of path directories and data input folder structures are completed, then the user can simply run the DL-SpiQA code. An output file labeled "DL-SpiQA Report.txt" will be generated at the specified dataPushPath directory containing a summary of QA results for each patient/treatment plan, including any flags raised due to overdosed or underdosed vertebral levels as well as detected vertebral volume discrepancies.
