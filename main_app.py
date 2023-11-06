import streamlit as st
import cv2
from skimage import io
from skimage.restoration import denoise_nl_means
import numpy as np
import time

# Define username-password pairs
USERS = {
    "admin": "Mydarlinghorse1986",
    "user2": "password2",
    "user3": "password3"
}

# Function to validate the username and password
def authenticate(username, password):
    return USERS.get(username) == password

# Function to upscale the image
def upscale_image(image, scale_factor):
    height, width, _ = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height))

# Function to denoise the image
def denoise_image(image, strength):
    return denoise_nl_means(image, h=strength)

# Function to remove colored backgrounds for PNG images
def remove_colored_background(image, background_color=(255, 255, 255), tolerance=30):
    lower_bound = np.array([background_color[0] - tolerance, background_color[1] - tolerance, background_color[2] - tolerance])
    upper_bound = np.array([background_color[0] + tolerance, background_color[1] + tolerance, background_color[2] + tolerance])

    mask = cv2.inRange(image, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)

    img_with_transparent_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    return img_with_transparent_bg

# Streamlit app with options for upscaling, denoising, and background removal
def main():
    st.title("Batch Picture Processing App")

    # Add username and password input fields
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", "", type="password")

    # Validate the username and password
    if username and password and authenticate(username, password):
        st.success(f"Authentication successful for user: {username}")

        scale_factor = st.slider("Select Upscaling Factor", min_value=1.0, max_value=4.0, step=0.1)
        denoise_strength = st.slider("Select Denoising Strength", min_value=0.01, max_value=0.5, step=0.01)
        remove_background = st.checkbox("Remove Colored Background (for PNG)")

        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        if uploaded_files and st.button("Process Images"):
            start_time = time.time()  # Record the start time
            progress_bar = st.progress(0)
            total_images = len(uploaded_files)
            processed_images = []

            for i, uploaded_file in enumerate(uploaded_files):
                original_image = io.imread(uploaded_file)
                upscaled_image = upscale_image(original_image, scale_factor)
                denoised_image = denoise_image(upscaled_image, denoise_strength)
                if remove_background:
                    processed_image = remove_colored_background(denoised_image)
                else:
                    processed_image = denoised_image
                processed_images.append(processed_image)

                # Update the progress bar
                progress_bar.progress(int((i + 1) / total_images * 100))

            end_time = time.time()  # Record the end time
            processing_time = end_time - start_time  # Calculate the total processing time

            st.header("Processed Images")
            for i, processed_image in enumerate(processed_images):
                st.image(processed_image, caption=f"Processed Image {i+1}", use_column_width=True)

            st.write(f"Time taken to process {total_images} images: {processing_time:.2f} seconds")
    elif username or password:
        st.error("Authentication failed. Please check your username and password.")

if __name__ == "__main__":
    main()
