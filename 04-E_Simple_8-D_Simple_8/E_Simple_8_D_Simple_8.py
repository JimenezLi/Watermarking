import os
from cv2 import cv2
from skimage import util
import shutil
import numpy as np
import random
import simple8_helper

alpha = np.sqrt(8)
# noise_standard_deviation = 140

dataset_path_root = 'E:/Datasets/vinbigdata/train'
generated_path_root = './cache/'
m = simple8_helper.get_messages()
w = simple8_helper.get_watermarks()
number_of_files = 200
file_list = list(os.listdir(dataset_path_root))[:number_of_files]

np.random.seed(random.randint(0, 99999))

if os.path.exists(generated_path_root):
    shutil.rmtree(generated_path_root)
os.makedirs(generated_path_root)


# ------------------------------------ Random Work --------------------------------------
def test_random_work():
    print('Testing random work')
    random_work_path = os.path.join(generated_path_root, 'random_work')
    if not os.path.exists(random_work_path):
        os.makedirs(random_work_path)

    # E_Simple_8 Embedding
    for file_index, file_name in enumerate(file_list):
        if file_index % 100 == 0:
            print(f'Embedding: Processed {file_index}/{len(file_list)} Files')

        img = cv2.imread(os.path.join(dataset_path_root, file_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.size, order='C')

        random_work_message = random.randint(0, 255)
        final_watermark = simple8_helper.get_final_watermark_normalized(random_work_message, w)
        new_img = img + alpha * final_watermark
        new_img = new_img.reshape(512, 512)
        cv2.imwrite(os.path.join(random_work_path, f'{file_index + 1}_-1_.png'), img)
        cv2.imwrite(os.path.join(random_work_path, f'{file_index + 1}_{random_work_message}_.png'), new_img)

    # D_SIMPLE_8 Detecting
    random_work_list = list(os.listdir(random_work_path))

    random_work_cases = number_of_files
    random_work_correct = 0
    random_work_false_positive = 0
    random_work_false_negative = 0

    for file_index, file_name in enumerate(random_work_list):
        if file_index % 100 == 0:
            print(f'Detecting: Processed {file_index}/{len(random_work_list)} Files')

        img = cv2.imread(os.path.join(random_work_path, file_name), cv2.IMREAD_GRAYSCALE)
        message = simple8_helper.decode(img, w)
        expected_message = int(file_name.split('_')[1])

        if expected_message != -1 and expected_message == message:
            random_work_correct += 1
        elif expected_message != -1 and message == -1:
            random_work_false_negative += 1
        elif expected_message == -1 and message != -1:
            random_work_false_positive += 1

    print('Random work cases:   ', random_work_cases)
    print('Correct cases:       ', random_work_correct, f'{random_work_correct * 100 / random_work_cases}%')
    print('False positive cases:', random_work_false_positive,
          f'{random_work_false_positive * 100 / random_work_cases}%')
    print('False negative cases:', random_work_false_negative,
          f'{random_work_false_negative * 100 / random_work_cases}%')
    print()


# ------------------------------------ Random Watermark ---------------------------------
def test_random_watermark():
    print('Testing random watermark')
    random_watermark_path = os.path.join(generated_path_root, 'random_watermark')
    random_watermark_file = file_list[0]
    random_watermark_count = 200
    random_watermarks = []
    if not os.path.exists(random_watermark_path):
        os.makedirs(random_watermark_path)

    # E_Simple_8 Embedding
    random_watermark_img = cv2.imread(os.path.join(dataset_path_root, random_watermark_file), cv2.IMREAD_GRAYSCALE)
    random_watermark_img = random_watermark_img.reshape(random_watermark_img.size, order='C')
    for watermark_index in range(random_watermark_count):
        if watermark_index % 100 == 0:
            print(f'Embedding: Processed {watermark_index}/{random_watermark_count} Files')

        watermark = simple8_helper.get_watermarks(random=True)
        random_watermarks.append(watermark)
        random_watermark_message = random.randint(0, 255)
        final_watermark = simple8_helper.get_final_watermark_normalized(random_watermark_message, watermark)
        new_img = random_watermark_img + alpha * final_watermark
        new_img = new_img.reshape(512, 512)
        cv2.imwrite(os.path.join(random_watermark_path, f'{random_watermark_message}_{watermark_index + 1}_.jpg'),
                    new_img)

    # D_SIMPLE_8 Detecting
    random_watermark_list = list(os.listdir(random_watermark_path))

    random_watermark_cases = number_of_files
    random_watermark_correct = 0
    random_watermark_false_positive = 0
    random_watermark_false_negative = 0

    for file_index, file_name in enumerate(random_watermark_list):
        if file_index % 100 == 0:
            print(f'Detecting: Processed {file_index}/{len(random_watermark_list)} Files')

        img = cv2.imread(os.path.join(random_watermark_path, file_name), cv2.IMREAD_GRAYSCALE)
        expected_message = int(file_name.split('_')[0])
        watermark = random_watermarks[int(file_name.split('_')[1]) - 1]
        message = simple8_helper.decode(img, watermark)

        if expected_message == message:
            random_watermark_correct += 1
        elif message == -1:
            random_watermark_false_negative += 1

    for watermark in random_watermarks:
        img = cv2.imread(os.path.join(dataset_path_root, random_watermark_file), cv2.IMREAD_GRAYSCALE)
        detected_message = simple8_helper.decode(img, watermark)
        if detected_message != -1:
            random_watermark_false_positive += 1

    print('Random watermark cases:', random_watermark_cases)
    print('Correct cases:         ', random_watermark_correct,
          f'{random_watermark_correct * 100 / random_watermark_cases}%')
    print('False positive cases:  ', random_watermark_false_positive,
          f'{random_watermark_false_positive * 100 / random_watermark_cases}%')
    print('False negative cases:  ', random_watermark_false_negative,
          f'{random_watermark_false_negative * 100 / random_watermark_cases}%')
    print()


# ------------------------------------ Longer Message -----------------------------------
def test_longer_message(message_length):
    print(f'Testing longer message {message_length}')
    longer_message_path = os.path.join(generated_path_root, f'longer_message_{message_length}')
    longer_message_watermark = simple8_helper.get_watermarks(message_bits=message_length)
    if not os.path.exists(longer_message_path):
        os.makedirs(longer_message_path)

    # E_Simple_8 Embedding
    for file_index, file_name in enumerate(file_list):
        if file_index % 100 == 0:
            print(f'Embedding: Processed {file_index}/{len(file_list)} Files')

        img = cv2.imread(os.path.join(dataset_path_root, file_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.size, order='C')

        longer_message_message = random.randint(0, 2 ** message_length - 1)
        final_watermark = simple8_helper.get_final_watermark_normalized(longer_message_message,
                                                                        longer_message_watermark,
                                                                        message_bits=message_length)
        new_img = img + alpha * final_watermark
        new_img = new_img.reshape(512, 512)
        cv2.imwrite(os.path.join(longer_message_path, f'{file_index + 1}_-1_.png'), img)
        cv2.imwrite(os.path.join(longer_message_path, f'{file_index + 1}_{longer_message_message}_.png'), new_img)

    # D_SIMPLE_8 Detecting
    longer_message_list = list(os.listdir(longer_message_path))

    longer_message_cases = number_of_files
    longer_message_correct = 0
    longer_message_false_positive = 0
    longer_message_false_negative = 0

    for file_index, file_name in enumerate(longer_message_list):
        if file_index % 100 == 0:
            print(f'Detecting: Processed {file_index}/{len(longer_message_list)} Files')

        img = cv2.imread(os.path.join(longer_message_path, file_name), cv2.IMREAD_GRAYSCALE)
        message = simple8_helper.decode(img, longer_message_watermark, message_bits=message_length)
        expected_message = int(file_name.split('_')[1])

        if expected_message != -1 and expected_message == message:
            longer_message_correct += 1
        elif expected_message != -1 and message == -1:
            longer_message_false_negative += 1
        elif expected_message == -1 and message != -1:
            longer_message_false_positive += 1

    print('Longer message cases:', longer_message_cases)
    print('Correct cases:       ', longer_message_correct, f'{longer_message_correct * 100 / longer_message_cases}%')
    print('False positive cases:', longer_message_false_positive,
          f'{longer_message_false_positive * 100 / longer_message_cases}%')
    print('False negative cases:', longer_message_false_negative,
          f'{longer_message_false_negative * 100 / longer_message_cases}%')
    print()


if __name__ == '__main__':
    test_longer_message(4)
    test_longer_message(16)
    test_longer_message(24)
    test_longer_message(32)

