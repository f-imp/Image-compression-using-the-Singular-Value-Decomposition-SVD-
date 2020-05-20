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


def perform_compression(image_path, k, output_path, output_name, color_space=None):
    img = cv2.imread(image_path)
    mode = ""
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_" + str(k)
    elif color_space is None or color_space != "grey":
        mode = "_compressed_" + str(k)
    compressed_image = compress_image(image=img, truncating_value=k)
    name = output_name + mode + ".jpg"
    cv2.imwrite(output_path + name, compressed_image)
    compression_ratio = round(k * (np.shape(img)[0] + np.shape(img)[1] + 1) / (np.shape(img)[0] * np.shape(img)[1]), 3)
    MSE = np.sum((img - compressed_image) ** 2) / float(img.shape[0] * img.shape[1])
    if color_space is not "grey":
        SSIM = structural_similarity(img, compressed_image, multichannel=True)
    else:
        SSIM = structural_similarity(img, compressed_image)
    return compressed_image, compression_ratio, MSE, SSIM


def optimal_compression_threshold(image_path, output_path, output_name, threshold=None, color_space=None):
    img = cv2.imread(image_path)
    mode = "_compressed_optimal_"
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_optimal_"
    if threshold is None:
        threshold = 70
    step = 10
    k = 1
    details = {'cut_off': [], 'ssim': []}
    flag = True
    c = 0
    while flag is True and k < np.minimum(np.shape(img)[0], np.shape(img)[1]) - step:
        compressed_image = compress_image(image=img, truncating_value=k)
        if color_space is not "grey":
            SSIM = structural_similarity(img, compressed_image, multichannel=True)
        else:
            SSIM = structural_similarity(img, compressed_image)
        details['cut_off'].append(k)
        details['ssim'].append(SSIM * 100)

        if SSIM * 100 < threshold:
            if c == 0:
                k += step - 1
            else:
                k += step
            c += 1
        else:
            flag = False
            name = output_name + mode + str(details['cut_off'][-1]) + ".jpg"
            cv2.imwrite(output_path + name, compressed_image)
    return compressed_image, details


def optimal_compression_relative_error(image_path, output_path, output_name, threshold=None,
                                       step=None, color_space=None):
    img = cv2.imread(image_path)
    mode = ""
    if color_space is "grey":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = "_grey_compressed_optimal_"
    elif color_space is None or color_space is not "grey":
        mode = "_compressed_optimal_"
    if threshold is None or threshold == 0:
        threshold = 0.01
    if step is None or step > min(np.shape(img)[0], np.shape(img)[1]):
        step = 10

    details = {'cut_off_sequence': [1, 1 + step - 1], 'ssim': [], 'relative_error': [0]}
    images = []

    compressed_image1 = compress_image(image=img, truncating_value=details['cut_off_sequence'][-2])
    images.append(compressed_image1)
    compressed_image2 = compress_image(image=img, truncating_value=details['cut_off_sequence'][-1])
    images.append(compressed_image2)
    if color_space is not "grey":
        SSIM1 = structural_similarity(img, compressed_image1, multichannel=True) * 100
        SSIM = structural_similarity(img, compressed_image2, multichannel=True) * 100
    else:
        SSIM1 = structural_similarity(img, compressed_image1) * 100
        SSIM = structural_similarity(img, compressed_image2) * 100
    details['ssim'].append(SSIM1)
    details['ssim'].append(SSIM)
    details['relative_error'].append(SSIM / 100 - SSIM1 / 100)
    while (details['ssim'][-1] / 100 - details['ssim'][-2] / 100) > threshold and details['cut_off_sequence'][
        -1] < np.minimum(np.shape(img)[0], np.shape(img)[1]) - step:
        details['cut_off_sequence'].append(details['cut_off_sequence'][-1] + step)
        compressed_image_iteration = compress_image(image=img, truncating_value=details['cut_off_sequence'][-1])
        images.append(compressed_image_iteration)
        if color_space is not "grey":
            SSIM_iteration = structural_similarity(img, compressed_image_iteration, multichannel=True) * 100
        else:
            SSIM_iteration = structural_similarity(img, compressed_image_iteration) * 100
        details['ssim'].append(SSIM_iteration)
        details['relative_error'].append(details['ssim'][-1] / 100 - details['ssim'][-2] / 100)
    name = output_name + mode + str(details['cut_off_sequence'][-1]) + ".jpg"
    cv2.imwrite(output_path + name, images[-1])
    return images[-1], details
