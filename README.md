# Rain Tracking
This repo contains details about training a detection algorithm for rain droplets, which feeds into a SORT tracking algorithm to count to number of rain droplets over time. The detection portion uses Retinanet with Resnet18, although there are remaining artifacts from a multitask model (actions), and inference is relatively slow because of that too. The best model achieved a mAP_train = 93.6, mAP_test = 80.0, and these numbers are robust to the few optimization strategies tested.

### Directory Setup
The following is the directory setup for using colab notebook:
```
├── data
│   ├── rain_all_data.csv, rain_train_data.csv, rain_val_data.csv, rain_classes.csv
├── logs
│       ├── best_model
│       ├── another_model
├── notebooks 
│       ├── xml_to_csv.ipynb, tRAIN.ipynb, infeRAINce.ipynb, tracking_rain.py
├── smooth_filter_utils (can use to further smooth/filter detections before tracking)
├── videos
```

### Data
A single video was taken from youtube, cropped to a 250px250px video, from which 110 frames were extracted over from 50s->80s, or every 10th frame in this window. LabelIMG was used to label 680 droplet contours, which were split 510 train 170 val. For training, a custom data augmentation pipeline was utilized to achieve higher accuracy. 

### Prerequisites
* Python (version >= 3.5)
* PyTorch (version 1.6.0)

### Results
<p align="center">
  <img width="200" height="200" src=rain_short.gif>
  <img width="286" height="200" src=forcast.jpg>
</p>

