import cv2
import numpy as np
from skimage.metrics import structural_similarity


def T_SVD(channel, k):
    U, S, V_t = np.linalg.svd(channel)
    Sigma = np.diag(S)
    U = U[:, :k]
    Sigma = Sigma[:k, :k]
    V_t = V_t[:k, :]
    return U, Sigma, V_t


def compress_image(image, truncating_value):
    channels = cv2.split(image)
    number_channels = np.shape(channels)[0]
    partial_results = []
    for i in range(number_channels):
        U, S, Vt = T_SVD(channel=channels[i], k=truncating_value)
        product = np.dot(np.dot(U, S), Vt)
        partial_results.append(product)
    partial_results = np.asarray(partial_results)
    compressed_image = cv2.merge(partial_results)
    return compressed_image


def perform_compression(image_path, percentage, output_path, output_name, color_space=None):
    img = cv2.imread(image_path)
    mode = ""
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_" + str(percentage)
    elif color_space is None or color_space != "grey":
        mode = "_compressed_" + str(percentage)

    k = int((min(np.shape(img)[0], np.shape(img)[1]) * percentage) / 100)
    compressed_image = compress_image(image=img, truncating_value=k)
    name = output_name + mode + ".jpg"
    cv2.imwrite(output_path + name, compressed_image)
    compression_ratio = round(k * (np.shape(img)[0] + np.shape(img)[1] + 1) / (np.shape(img)[0] * np.shape(img)[1]), 3)
    MSE = np.sum((img - compressed_image) ** 2) / float(img.shape[0] * img.shape[1])
    if color_space is not "grey":
        SSIM = structural_similarity(img, compressed_image, multichannel=True)
    else:
        SSIM = structural_similarity(img, compressed_image)
    return compressed_image, k, compression_ratio, MSE, SSIM


def optimal_compression_threshold(image_path, output_path, output_name, threshold_similarity=None, color_space=None):
    img = cv2.imread(image_path)
    mode = "_compressed_optimal_"
    min_dimension = min(np.shape(img)[0], np.shape(img)[1])
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_optimal_"
    if threshold_similarity is None:
        threshold_similarity = 70
    step = 5
    percentage = 1
    details = {'percentage': [], 'cut_off': [], 'ssim': []}
    flag = True
    c = 0
    while flag is True and percentage <= 100:
        k = int((min_dimension / 100) * percentage)
        compressed_image = compress_image(image=img, truncating_value=k)
        if color_space is not "grey":
            SSIM = structural_similarity(img, compressed_image, multichannel=True)
        else:
            SSIM = structural_similarity(img, compressed_image)
        details['percentage'].append(percentage)
        details['cut_off'].append(k)
        details['ssim'].append(round(SSIM * 100, 2))

        if SSIM * 100 < threshold_similarity:
            if c == 0:
                percentage += step - 1
            else:
                percentage += step
            c += 1
        else:
            flag = False
        name = output_name + mode + str(details['percentage'][-1]) + ".jpg"
        cv2.imwrite(output_path + name, compressed_image)
    return compressed_image, details


def optimal_compression_relative_error(image_path, output_path, output_name, threshold_similarity=None,
                                       step=None, color_space=None):
    img = cv2.imread(image_path)
    mode = ""
    min_dimension = min(np.shape(img)[0], np.shape(img)[1])
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_optimal_"
    elif color_space is None or color_space is not "grey":
        mode = "_compressed_optimal_"
    if threshold_similarity is None:
        threshold_similarity = 0.1
    if step is None or step > 100 or step <= 0:
        step = 10

    details = {'percentage': [1, step],
               'cut_off': [int(min_dimension / 100 * 1), int(min_dimension / 100 * step)],
               'ssim': [],
               'relative_error': [0]}
    images = []

    compressed_image1 = compress_image(image=img,
                                       truncating_value=details['cut_off'][-2])
    name = output_name + mode + str(details['percentage'][-2]) + ".jpg"
    cv2.imwrite(output_path + name, compressed_image1)
    images.append(compressed_image1)
    compressed_image2 = compress_image(image=img,
                                       truncating_value=details['cut_off'][-1])
    name = output_name + mode + str(details['percentage'][-1]) + ".jpg"
    cv2.imwrite(output_path + name, compressed_image2)
    images.append(compressed_image2)
    if color_space is not "grey":
        SSIM1 = structural_similarity(img, compressed_image1, multichannel=True) * 100
        SSIM = structural_similarity(img, compressed_image2, multichannel=True) * 100
    else:
        SSIM1 = structural_similarity(img, compressed_image1) * 100
        SSIM = structural_similarity(img, compressed_image2) * 100
    details['ssim'].append(round(SSIM1, 2))
    details['ssim'].append(round(SSIM, 2))
    details['relative_error'].append(SSIM / 100 - SSIM1 / 100)
    while details['relative_error'][-1] > threshold_similarity and details['percentage'][-1] < (100 - step):
        details['percentage'].append(details['percentage'][-1] + step)
        details['cut_off'].append(int(min_dimension / 100 * (details['percentage'][-1])))
        compressed_image_iteration = compress_image(image=img, truncating_value=details['cut_off'][-1])
        images.append(compressed_image_iteration)
        if color_space is not "grey":
            SSIM_iteration = structural_similarity(img, compressed_image_iteration, multichannel=True) * 100
        else:
            SSIM_iteration = structural_similarity(img, compressed_image_iteration) * 100
        details['ssim'].append(round(SSIM_iteration, 2))
        details['relative_error'].append(details['ssim'][-1] / 100 - details['ssim'][-2] / 100)
        name = output_name + mode + str(details['percentage'][-1]) + ".jpg"
        cv2.imwrite(output_path + name, images[-1])
    return images[-1], details
