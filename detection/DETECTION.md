# README
This file provides instructions on how to perform instructions upon `BOP19 Dataset`

## Setup
1. Set up `mmdetection` baseline according to [here](https://github.com/open-mmlab/mmdetection). 
2. Download all testsets in `detection/bop19` from [here](https://bop.felk.cvut.cz/datasets/)(or you could download somewhere else and set up a soft link).
    <br> Take `lm` as an example
    ```
    export SRC=http://ptak.felk.cvut.cz/6DB/public/bop_datasets
    wget $SRC/lm_base.zip        # Base archive with dataset info, camera parameters, etc.
    wget $SRC/lm_models.zip      # 3D object models.
    wget $SRC/lm_test_all.zip    # All test images ("_bop19" for a subset used in the BOP Challenge 2019).
    wget $SRC/lm_train.zip       # Training images.

    unzip lm_base.zip            # Contains folder "lm".
    unzip lm_models.zip -d lm    # Unpacks to "lm".
    unzip lm_test_all.zip -d lm  # Unpacks to "lm".
    unzip lm_train.zip -d lm     # Unpacks to "lm".
    ```
    <b>Note: </b> All datasets should be in `detection/bop19` whose file system should be as following:
    ```
    * lmo
        * test
            * <scene_id>
                * rgb
                    * <img_id>.png
                    * ...
                * mask_visib
                * depth
                * ...
    * ycbv
        * test
            * ...
    * ...
    ```
3. Find the root directory of your `mmdetction` path
    ```
    # Please specify the path below
    export MMDETECTION_PATH=<your mmdetection path>
    export DETECTION_PATH=<detection>
    
    # Set up soft link here to self-define the bop19-dataset
    cd ${MMDETECTION_PATH}/mmdet/
    ln -s ${DETECTION_PATH}/datasets

    # return to your work place
    cd ${DETECTION_PATH}
    ```

4. Make Sure you have all models in `${DETECTION_PATH}/models`
5. Generate the detection results by
    ```
    pip install tqdm -y
    python inference.py <dataset name>
    ```
6. Then you have your detection results in `${DETECTION_PATH}/out`, and the file system is as below:
    ```
    * hb
        * hb_<scene_id>_detection_result.json
        * ...
    * ...
    ```
    You can go on test on the Pose Part using these files.