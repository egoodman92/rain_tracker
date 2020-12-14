# Rain Tracking
This repo contains details about training a detection algorithm for rain droplets, which feeds into a SORT tracking algorithm to count to number of rain droplets over time. The detection portion uses Retinanet with Resnet18, although there are remaining artifacts from a multitask model (actions), and inference is relatively slow because of that too. The best model achieved a mAP_train = 85.3, mAP_test = 82.1, and these numbers are robust to the few optimization changes tested.

### Directory Setup

The following is the directory setup for using colab notebook:
```
├── data
│   ├── rain_all_data.csv, rain_train_data.csv, rain_val_data.csv, rain_classes.csv
├── logs
│       ├── model1
│       ├── model2
├── notebooks 
│       ├── xml_to_csv.ipynb, tRAIN.ipynb, infeRAINce.ipynb, tracking_rain.py
├── smooth_filter_utils (can use to further smooth/filter detections before tracking)
├── videos
```

### Prerequisites
* Python (version >= 3.5)
* PyTorch (version 1.6.0)
