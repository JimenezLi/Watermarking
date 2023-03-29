import os
import cv2.cv2 as cv2
import numpy as np

cache_path = '../cache/0'
data_path = '../data'
default_seed = 20230324
default_threshold = 0.22
default_beta = max(1 - default_threshold, 0.3)
image_shape = (512, 512)
NO_WATERMARK = -1


if not os.path.exists(data_path):
    raise FileNotFoundError(f"No image found in {os.path.join(os.getcwd(), data_path)}")

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


def generate_normalized_watermark(seed=None, image_shape=image_shape):
    """
    :return: A watermark of zero mean and unit variance.
    """
    np.random.seed(seed)
    watermark = np.random.random(image_shape)
    normalized_watermark = (watermark - np.mean(watermark)) / np.sqrt(np.var(watermark))
    return normalized_watermark


default_watermark = generate_normalized_watermark(default_seed)
default_image_shape = image_shape


def E_FIXED_LC_getalpha(image, watermark, threshold, beta):
    return (image.size * (threshold + beta) - np.vdot(image, watermark)) / (np.vdot(watermark, watermark))


def E_FIXED_LC(file_name, message, watermark=default_watermark, image_shape=default_image_shape,
               threshold=default_threshold, beta=default_beta, notes=None,
               input_path=data_path, output_path=cache_path):
    """
    :param file_name: The file name in data_path; for example, 'more_data_0.jpg'
    :param message: The message to be encoded into image
    :param threshold: Detection threshold for D_LC algorithm; may depend on the dataset
    :param beta: beta>0 is an embedding strength parameter, to ensure the magnitude of the detection strength is greater
than the threshold
    :param notes: Extra information; left blank by default
    :return: None
    """
    assert beta > 0

    img = cv2.imread(os.path.join(input_path, file_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
    cut_img = img[0:image_shape[0], 0:image_shape[1]]
    assert cut_img.shape == image_shape == watermark.shape

    if message == 1:
        encoded_img = cut_img + E_FIXED_LC_getalpha(cut_img, watermark, threshold, beta) * watermark
    elif message == 0:
        encoded_img = cut_img - E_FIXED_LC_getalpha(cut_img, -watermark, threshold, beta) * watermark
    else:
        encoded_img = cut_img
        message = NO_WATERMARK
    notes = str() if notes is None else (str(notes) + '_')

    # Output
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv2.imwrite(os.path.join(output_path, f'{message}_{notes}{file_name}'), encoded_img)


def D_LC(file_name, watermark=default_watermark, image_shape=default_image_shape, mode='accept',
         threshold=default_threshold, input_path=cache_path):
    """
    :param file_name: The file name in data_path; for example, 'more_data_0.jpg'
    :param mode: Chosen from 'zlc' and 'accept'; zlc mode returns the linear correlation value,
     and accept mode returns the decoded message.
    :param threshold: The message detection value; must be given a value in accept mode.
    :return: See mode in params.
    """
    assert mode in ['zlc', 'accept']
    encoded_img = cv2.imread(os.path.join(input_path, file_name), cv2.IMREAD_GRAYSCALE)
    if encoded_img is None:
        raise FileNotFoundError
    assert encoded_img.shape == image_shape == watermark.shape

    zlc = np.vdot(encoded_img, watermark) / watermark.size
    if mode == 'zlc':
        return zlc
    elif mode == 'accept':
        if zlc > threshold:
            return 1
        elif zlc < -threshold:
            return 0
        else:
            return NO_WATERMARK


def get_blackwhite_pixel_rate(img_name, threshold, image_shape=default_image_shape, data_path=data_path):
    img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
    img = img[0:image_shape[0], 0:image_shape[1]]
    img = np.reshape(img, img.size, order='C')
    blackwhite_pixel_rate = (np.count_nonzero(img < threshold) + np.count_nonzero(img > 255 - threshold)) / img.size
    return blackwhite_pixel_rate


if __name__ == '__main__':
    E_FIXED_LC('more_data_0.jpg', 0)
    E_FIXED_LC('more_data_0.jpg', 1)
    E_FIXED_LC('more_data_0.jpg', NO_WATERMARK)

    threshold = 20
    print('Blackwhite pixel rate of more_data_0.jpg:', get_blackwhite_pixel_rate('more_data_0.jpg', threshold))
    print('Blackwhite pixel rate of brain1.bmp:', get_blackwhite_pixel_rate('brain1.bmp', threshold, image_shape=(256, 256)))

    print('Encoding 1:', D_LC('1_more_data_0.jpg', mode='zlc'))
    print('Encoding 0:', D_LC('0_more_data_0.jpg', mode='zlc'))
    print('No watermark:', D_LC('-1_more_data_0.jpg', mode='zlc'))

    print('Encoding 1 Message:', D_LC('1_more_data_0.jpg', mode='accept'))
    print('Encoding 0 Message:', D_LC('0_more_data_0.jpg', mode='accept'))
    print('No watermark Message:', D_LC('-1_more_data_0.jpg', mode='accept'))
