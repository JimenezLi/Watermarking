import numpy as np
from math import sqrt

seed = 20230321
bits = 8
size = 512 * 512
messages = [94, 155, 9, 187, 229, 49, 225, 89, 162, 236]
threshold = 0.005


def get_seed():
    return seed


def get_message_length():
    return bits


def get_watermarks(message_bits=bits, image_size=size, random=False):
    lst = []
    if not random:
        np.random.seed(get_seed())
    for i in range(message_bits):
        lst.append(normalized(np.random.random(image_size)))
    return lst[:]


def normalized(array):
    return (array - np.mean(array)) / np.sqrt(np.var(array))


def get_messages():
    return messages[:]


def get_final_watermark_normalized(message, watermark_list, message_bits=bits, image_size=size):
    assert len(watermark_list) == message_bits

    watermark = np.zeros(image_size)
    message_string = (message_bits - len(bin(message)[2:])) * '0' + bin(message)[2:]
    for i in range(len(message_string)):
        if message_string[i] == '1':
            watermark += watermark_list[i]
        else:
            watermark -= watermark_list[i]
        message /= 2
    watermark /= sqrt(message_bits)
    return watermark


def decode(image, watermark_list, message_bits=bits, image_size=size):
    image_array = image.reshape(image.size, order='C')
    assert len(watermark_list) == message_bits
    assert len(image_array) == image_size

    no_watermark_count = 0
    message = 0
    for i in range(message_bits):
        z_lc = np.corrcoef(image_array, watermark_list[i])[0][1]
        if z_lc > 0:
            message += 2 ** (message_bits - 1 - i)
        else:
            message += 0
        if threshold >= z_lc >= -threshold:
            no_watermark_count += 1
    return message if no_watermark_count <= bits // 2 else -1
