import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter
import matplotlib.pyplot as plt
import io

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Deteksi Lesi dan Klasifikasi Citra Menggunakan GLCM + KNN")

excel_file = st.file_uploader("Upload File Excel Label", type=['xlsx'])
image_files = st.file_uploader("Upload Semua Gambar (.jpg, .png)", type=['jpg', 'png'], accept_multiple_files=True)

if excel_file and image_files:
    img_df = pd.read_excel(excel_file)
    labeled_images = defaultdict(list)
    img_dict = {img.name: img for img in image_files}

    for idx, row in img_df.iterrows():
        filename = row['File']
        label = row['Type']
        if filename in img_dict:
            try:
                image = Image.open(img_dict[filename]).convert('RGB')
                labeled_images[label].append((filename, np.array(image)))
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")

    st.header("Visualisasi Gambar dan Label")
    for label, images in labeled_images.items():
        if images:
            st.subheader(f"Label: {label}")
            st.image(images[0][1], caption=images[0][0], use_column_width=True)

    def rgb_to_grayscale_manual(image_array):
        return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    def extract_features(image_array):
        gray = rgb_to_grayscale_manual(image_array)
        gray_uint8 = np.uint8(gray)
        glcm = graycomatrix(gray_uint8, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return [contrast, dissimilarity, homogeneity, energy, correlation]

    st.header("Deteksi Lesi dan Bounding Box")
    for label, items in labeled_images.items():
        for name, img in items[:1]:
            gray = rgb_to_grayscale_manual(img)
            binary = gray > gray.mean()
            labeled = label(binary)
            fig, ax = plt.subplots()
            ax.imshow(img)
            for region in regionprops(labeled):
                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            ax.set_title(f"{label} - {name}")
            st.pyplot(fig)

    st.header("Ekstraksi Fitur GLCM & Klasifikasi KNN")
    features = []
    labels = []
    for label, items in labeled_images.items():
        for name, img in items:
            feat = extract_features(img)
            features.append(feat)
            labels.append(label)

    if features:
        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Akurasi Model: {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Tidak ada fitur yang diekstrak.")
