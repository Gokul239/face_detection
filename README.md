Face Recognition App
This is a face recognition application that leverages deep learning models to identify faces in images and video streams using the ViT (Vision Transformer) architecture. It utilizes dlib for face detection, OpenCV for video processing, and LangChain for managing vector storage.

Features
Upload images to train the model with specific faces.
Identify faces in real-time using a webcam.
Utilize a vector database to store and retrieve face embeddings.
Display bounding boxes and labels around detected faces.
Requirements
To run this application, you'll need to have the following Python libraries installed:

dlib
opencv-python
torch
streamlit
transformers
scikit-learn
langchain
matplotlib
Pillow
numpy

Use requirements.txt file to install dependencies


Directory Structure
plaintext

.
├── main.py                  # Main application code
├── chroma_db               # Directory for Chroma vector database
└── Training_images/        # Directory for storing training images

Usage
Run the application:

Start the Streamlit server by executing the following command in your terminal:

streamlit run main.py
Upload Images:

Navigate to the "Image Input" tab.
Upload images of faces you want to recognize.
Enter a label for each uploaded image and click "Generate."
Webcam Identification:

Switch to the "Webcam Video" tab.
Click the "Start" button to begin face identification in real-time through your webcam.
Stop the Webcam:

Click the "Stop" button to stop the webcam feed.
Code Overview
Face Detection: Utilizes dlib to detect faces in images and video frames.
Embedding Generation: Uses ViTModel from Hugging Face Transformers to generate embeddings for detected faces.
Vector Database: Employs Chroma to store and query face embeddings.
User Interface: Built with Streamlit for an interactive experience.
Main Functions
face_detector(gray_image, img): Detects faces in a given image.
image_embedding(image): Generates embeddings for a detected face image.
training_images_finc(image, names): Loads images and stores their embeddings in the vector database.
video_idetification(frame_image): Identifies faces in video frames.
webcam_func(): Manages the webcam feed for real-time face identification.
tain_images(): Handles image uploads and training.
main(): Sets up the Streamlit app with tabs for image input and webcam functionality.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
The Hugging Face team for the Transformers library.
The dlib library for robust face detection.