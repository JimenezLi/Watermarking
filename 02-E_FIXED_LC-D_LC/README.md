# E_FIXED_LC/D_LC System
# WARNING: Do not use JPEG! JPEG has a huge loss when saving and loading.
# File Structure
```
.
│  README.md
│  requirements.txt
│
├─cache
│      <Generated when running>
│
├─data
│      <Default data path>
│
└─src
      test_random_watermark.py
      test_random_work.py
      watermarking.py

```
# Build & Run
## Build
My Python version is 3.9.11. You may also use any compatible Python.
```commandline
pip install -r requirements.txt
```
## Run
Before running, make sure you have specified the `data` folder path (default path is `./data`).
```commandline
cd ./src
python test_random_work.py
python test_random_watermark.py
```