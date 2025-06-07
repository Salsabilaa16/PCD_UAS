import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import cv2
import io

st.title("Detection Cancer Serviks")

# File Uploads
excel_file = st.file_uploader("Upload Excel Metadata (.xlsx)", type=["xlsx"])
image_files = st.file_uploader("Upload Image Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if excel_file and image_files:
    df = pd.read_excel(excel_file)
    image_dict = {img.name: img for img in image_files}
    labeled_images = defaultdict(list)

    for _, row in df.iterrows():
        filename, label = row['File'], row['Type']
        if filename in image_dict:
            image = Image.open(image_dict[filename])
            labeled_images[label].append(image)

    def rgb_to_grayscale_manual(image_array):
        return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    def median_filter(img, kernel_size=3):
        img = np.array(img)
        h, w = img.shape
        offset = kernel_size // 2
        padded = np.pad(img, offset, mode='edge')
        filtered = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                window = padded[i:i+kernel_size, j:j+kernel_size].flatten()
                filtered[i, j] = np.median(window)
        return filtered

    def histogram_equalization(gray_img):
        hist, bins = np.histogram(gray_img.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        return cdf_normalized[gray_img.astype(int)].astype('uint8')

    def apply_adaptive_thresholding(img, block_size=11, C=2):
        pad = block_size // 2
        padded = np.pad(img, pad, mode='reflect')
        result = np.zeros_like(img, dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                local = padded[i:i+block_size, j:j+block_size]
                threshold = np.mean(local) - C
                result[i, j] = 0 if img[i, j] > threshold else 255
        return result

    def erode(img, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(img, pad, mode='constant', constant_values=255)
        output = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                output[i, j] = 0 if np.all(region[kernel==1]==0) else 255
        return output

    def dilate(img, kernel):
        pad = kernel.shape[0] // 2
        padded = np.pad(img, pad, mode='constant', constant_values=255)
        output = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                output[i, j] = 0 if np.any(region[kernel==1]==0) else 255
        return output

    def closing_opening(img, kernel):
        return dilate(erode(dilate(img, kernel), kernel), kernel)

    kernel = np.ones((3,3), dtype=int)
    processed_images = defaultdict(list)
    features_list = []

    for label, imgs in labeled_images.items():
        for img in imgs:
            gray = rgb_to_grayscale_manual(np.array(img)).astype('uint8')
            median = median_filter(gray)
            equalized = histogram_equalization(median)
            adaptive = apply_adaptive_thresholding(equalized, block_size=25, C=3)
            morph = closing_opening(adaptive, kernel)

            for _ in range(2): morph = dilate(erode(morph, kernel), kernel)

            glcm = graycomatrix(morph, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features = {
                'label': label,
                'contrast': graycoprops(glcm, 'contrast')[0, 0],
                'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
                'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': graycoprops(glcm, 'energy')[0, 0],
                'correlation': graycoprops(glcm, 'correlation')[0, 0],
                'ASM': graycoprops(glcm, 'ASM')[0, 0],
            }
            features_list.append(features)

            labeled = (morph > 0).astype(np.uint8)
            _, labeled_img = cv2.connectedComponents(labeled)
            img_rgb = np.stack([gray]*3, axis=-1)
            for region in regionprops(labeled_img):
                if region.area > 300:
                    minr, minc, maxr, maxc = region.bbox
                    rr, cc = rectangle_perimeter((minr, minc), end=(maxr-1, maxc-1), shape=img_rgb.shape)
                    img_rgb[rr, cc] = [255, 0, 0]

            fig, ax = plt.subplots(1, 2, figsize=(10,4))
            ax[0].imshow(gray, cmap='gray')
            ax[0].set_title("Grayscale")
            ax[0].axis('off')
            ax[1].imshow(img_rgb)
            ax[1].set_title("Lesion Detection")
            ax[1].axis('off')
            st.pyplot(fig)

    st.subheader("GLCM Features")
    st.dataframe(pd.DataFrame(features_list))
