# VLEM

## Getting Started

### Construct Image

We conduct experiments on three eye movement datasets ZuCo, SB-SAT, CopCo.

The raw data can be found at:

**(1)** ZuCo: [https://osf.io/q3zws/](https://osf.io/q3zws/)

**(2)** SB-SAT: [https://github.com/ahnchive/SB-SAT](https://github.com/ahnchive/SB-SAT)

**(3)** CopCo: [https://osf.io/ud8s5/](https://osf.io/ud8s5/)

Follow these two steps to create the images for each dataset:

1. Get the raw data, unzip them, and put the files in `dataset` folder.
2. Run the following commands in turn to create the images:
   1. `cd dataset/zuco/process_script`, `python ConstructImage.py`
   2. `cd dataset/SB-SAT/process_script`, `python ConstructImage.py`
   3. `cd dataset/CopCo/process_script`, `python ConstructImage.py`

### Training

Before started, remember to insert your project directory into the system path by replacing the PROJECT_PATH in `sys.path.insert(0, 'PROJECT_PATH/code')` with your own project path.

```
python run_VisionTextRepresent.py
```

