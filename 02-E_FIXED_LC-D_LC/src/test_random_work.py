import os
import numpy as np
import watermarking
import cv2.cv2 as cv2
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Create Image List
    image_list_length = 200
    image_list = list(filter(lambda x: x.endswith('.jpg'), os.listdir('../data')))[:image_list_length]

    # Encode
    test_cache_path = '../cache/random_work'
    for img_index, img_name in enumerate(image_list):
        watermarking.E_FIXED_LC(img_name, 0, output_path=test_cache_path)
        watermarking.E_FIXED_LC(img_name, 1, output_path=test_cache_path)
        watermarking.E_FIXED_LC(img_name, -1, output_path=test_cache_path)
        print(f"\rEncoding image {img_index + 1}/{image_list_length}", end="", flush=True)
    print("\rEncoding complete", flush=True)

    # Decode
    encoded_image_list = os.listdir(test_cache_path)
    assert len(encoded_image_list) == image_list_length * 3
    image_1_list = []
    image_0_list = []
    image_nowatermark_list = []
    for img_name in encoded_image_list:
        message = int(img_name.split('_')[0])
        if message == 1:
            image_1_list.append(watermarking.D_LC(img_name, mode='zlc', input_path=test_cache_path))
        elif message == 0:
            image_0_list.append(watermarking.D_LC(img_name, mode='zlc', input_path=test_cache_path))
        else:
            image_nowatermark_list.append(watermarking.D_LC(img_name, mode='zlc', input_path=test_cache_path))

    x = np.linspace(-1, 1, 51)
    y1 = pd.cut(image_1_list, x, labels=x[:-1])
    y0 = pd.cut(image_0_list, x, labels=x[:-1])
    yn = pd.cut(image_nowatermark_list, x, labels=x[:-1])

    counts1 = pd.value_counts(y1, sort=False)
    counts0 = pd.value_counts(y0, sort=False)
    countsn = pd.value_counts(yn, sort=False)

    # Plot
    plt.xlabel('Detection value')
    plt.ylabel('Percentage of images')

    plt.plot(counts1.index, counts1 * (100 / image_list_length), label='m=1')
    plt.plot(counts0.index, counts0 * (100 / image_list_length), label='m=0')
    plt.plot(countsn.index, countsn * (100 / image_list_length), label='No watermark')
    plt.legend()
    plt.show()

    # Calculate rates
    threshold = watermarking.default_threshold
    false_positive_count = 0
    false_negative_count = 0

    for i in image_0_list:
        if i >= -threshold:
            false_negative_count += 1
    for i in image_1_list:
        if i <= threshold:
            false_negative_count += 1
    for i in image_nowatermark_list:
        if i > threshold or i < -threshold:
            false_positive_count += 1

    print('False positive rate:', false_positive_count / len(image_nowatermark_list))
    print('False negative rate:', false_negative_count / (len(image_0_list) + len(image_1_list)))
