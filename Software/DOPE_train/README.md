# DOPE Training

### 1. Creating Dataset

in the folder `pose_generation`, copy the `pose_generation_pallet.py` and paste your own Isaac Sim pose generation path( ex) C:\isaacsim\standalone_examples\replicator\pose_generation\) 



If you run `pose_generation_pallet.py`, there will be a dataset file `C:\Isaac_Data\Pallet_Dataset_warehouse_background`.



### 2. Train

run training code

```
cd path/to/train.py
python train.py
```



change the `DATA_PATH` and `OUTPUT_PATH`

```python
DATA_PATH   = r"C:\Isaac_Data\Pallet_Dataset_Total" 
OUTPUT_PATH = r"C:\DOPE_Training\train_test_final_251207"
```



### 3. Demo

Test your model with `demo.py`

```
cd path/to/demo.py
python demo.py
```



Change the Paths

```python
# PATHS
MODEL_PATH = r"C:\DOPE_Training\train_test_final_251207\best_model_multistage.pth"
INPUT_IMAGE = r"C:\isaacsim\FinalProject\pose_compare_output\01_yolo_bbox.png"
INPUT_JSON  = r"C:\Isaac_Data\Pallet_Dataset_Total\003140.json"

SAVE_VIS          = r"C:\DOPE_Training\train_test3\pred_vis_lines.png"          # GT + Pred keypoints
SAVE_DOPE_ORIENT  = r"C:\DOPE_Training\train_test3\pred_dope_orientation.png"   # DOPE 6D pose cuboid

SAVE_HEATMAP_DIR = r"C:\DOPE_Training\train_test3\heatmaps"
```

