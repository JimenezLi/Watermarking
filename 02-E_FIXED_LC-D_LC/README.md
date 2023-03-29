# E_FIXED_LC/D_LC System

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
│      <Place your images here>
│      <You can also use soft link to 01-E_BLIND-D_LC/data>
│      
└─src
      test_random_watermark.py
      test_random_work.py
      watermarking.py

```
# Build & Run
## Build
```commandline
pip install -r requirements.txt
```
## Run
```commandline
python test_random_work.py
python test_random_watermark.py
```