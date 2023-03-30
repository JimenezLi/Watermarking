import os

import cv2
import numpy as np
import watermarking
import pandas as pd
import matplotlib.pyplot as plt

# single_image = 'boat.bmp'
# single_image = 'rec_enlarged.bmp'
single_image = 'brain1.BMP'


if __name__ == '__main__':
    # Create Watermark List
    np.random.seed(watermarking.default_seed)
    watermark_count = 200
    watermark_seed_list = np.random.random_integers(0, 2147483647, watermark_count)
    watermark_list = [watermarking.generate_normalized_watermark(i) for i in watermark_seed_list]

    # Encode
    test_cache_path = '../cache/random_watermark'
    for watermark_index, watermark in enumerate(watermark_list):
        watermarking.E_BLIND(single_image, 0, watermark=watermark, notes=watermark_index, output_path=test_cache_path)
        watermarking.E_BLIND(single_image, 1, watermark=watermark, notes=watermark_index, output_path=test_cache_path)
        watermarking.E_BLIND(single_image, -1, watermark=watermark, notes=watermark_index, output_path=test_cache_path)
        print(f"\rEncoding watermark {watermark_index + 1}/{watermark_count}", end="", flush=True)
    print("\rEncoding complete", flush=True)

    # Decode
    encoded_image_list = list(filter(lambda x: x.endswith(single_image), os.listdir(test_cache_path)))
    assert len(encoded_image_list) == watermark_count * 3
    image_1_list = []
    image_0_list = []
    image_no_list = []
    for img_name in encoded_image_list:
        message, watermark_index = int(img_name.split('_')[0]), int(img_name.split('_')[1])
        if message == 1:
            image_1_list.append(watermarking.D_LC(img_name, mode='zlc', watermark=watermark_list[watermark_index], input_path=test_cache_path))
        elif message == 0:
            image_0_list.append(watermarking.D_LC(img_name, mode='zlc', watermark=watermark_list[watermark_index], input_path=test_cache_path))
        else:
            image_no_list.append(watermarking.D_LC(img_name, mode='zlc', watermark=watermark_list[watermark_index], input_path=test_cache_path))

    x = np.linspace(-2, 2, 51)
    y1 = pd.cut(image_1_list, x, labels=x[:-1])
    y0 = pd.cut(image_0_list, x, labels=x[:-1])
    yn = pd.cut(image_no_list, x, labels=x[:-1])

    counts1 = pd.value_counts(y1, sort=False)
    counts0 = pd.value_counts(y0, sort=False)
    countsn = pd.value_counts(yn, sort=False)

    # Plot
    plt.xlabel('Detection value')
    plt.ylabel('Percentage of images')

    plt.plot(counts1.index, counts1 * (100 / watermark_count), label='m=1')
    plt.plot(counts0.index, counts0 * (100 / watermark_count), label='m=0')
    plt.plot(countsn.index, countsn * (100 / watermark_count), label='No watermark')
    plt.legend()
    plt.show()

    # Calculate rates
    threshold = 0.7
    false_positive_count = 0
    false_negative_count = 0

    for i in image_0_list:
        if i >= -threshold:
            false_negative_count += 1
    for i in image_1_list:
        if i <= threshold:
            false_negative_count += 1
    for i in image_no_list:
        if i > threshold or i < -threshold:
            false_positive_count += 1

    print('Black white pixel rate:', watermarking.get_blackwhite_pixel_rate(single_image, 2))
    print('False positive rate:', false_positive_count / len(image_no_list))
    print('False negative rate:', false_negative_count / (len(image_0_list) + len(image_1_list)))
