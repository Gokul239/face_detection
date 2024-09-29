import dlib
import cv2
import torch
import streamlit as st
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import Chroma
from langchain.schema import Document
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')


#Font, styles of rectangle
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
color = (0, 255, 0)  # Green color in BGR
thickness = 10
lineType = cv2.LINE_AA


#Vector db directory
persist_directory = "./chroma_db"

#Create image directory
image_directory = "./Training_images/"
os.makedirs(save_directory, exist_ok=True)
# Load the pre-trained ViT model and feature extractor
detector = dlib.get_frontal_face_detector()
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

#Loading Chrom vector db
vector_store = Chroma(persist_directory=persist_directory  , embedding_function=None)
tab1, tab2 = st.tabs(["Image Input", "Webcam Video"])
#Face detector function
def face_detector(gray_image, img):
    # Detect faces
    faces = detector(gray_image)

    face_region = []
        # Draw rectangles around faces
    for i, face in enumerate(faces):
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_region.append(img[y:y + h, x:x + w])
    return face_region, faces


#Convert Face Image to embedding function
def image_embedding(image):
        inputs = feature_extractor(images=image, return_tensors='pt')
        # Perform inference to get embeddings
        with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        # Extract the embedding for the [CLS] token
        cls_embedding = embeddings[:, 0, :]  # The first token represents the image embedding

        # Convert to numpy array if needed
        embedding_array = np.array(cls_embedding)
        
        return embedding_array

#Loading images and storing in vector db
def training_images_finc(image, names):
        img = cv2.imread(image)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_region, face_region_box = face_detector(gray_image, img)
        if len(face_region) >0:
            known_image_embedding_1 = [image_embedding(i).tolist()[0] for i in face_region]
            known_names_2 = [Document(page_content=names) for i in known_image_embedding_1]
            known_names_1 = [names for i in range(len(known_names_2))]
            ids = [(f'{names}{i}') for i in range(len(known_image_embedding_1))]
            vector_store._collection.add( ids = ids, embeddings=known_image_embedding_1, documents=known_names_1)
    

#Calling function




#Predicting face from video frame function
def video_idetification(frame_image):
    u_grey = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
    un_face_region, unknow_region = face_detector(u_grey, frame_image)
    un_known_image_embedding = [image_embedding(i) for i in un_face_region]
    for  i, u_embed in enumerate(un_known_image_embedding):
            results_with_scores = vector_store._collection.query(
                    query_embeddings=u_embed.tolist(),      
                    n_results=1,                             
                    include=["embeddings", 'documents']                    
                    )
            similarity_score = cosine_similarity(results_with_scores['embeddings'][0], u_embed.tolist())
            if  similarity_score[0][0]> 0.5:
                x, y, w, h = (unknow_region[i].left(), unknow_region[i].top(), unknow_region[i].width(), unknow_region[i].height())
                cv2.rectangle(frame_image, (x, y), (x + w, y + h), (0, 0, 255), 4)
                cv2.putText(frame_image, results_with_scores['documents'][0][0], (x, y-60), font, fontScale, color, thickness, lineType)
    return frame_image

#Main function to enable webcam
def webcam_func():
    # Open the default camera (usually the webcam)
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    button_label = "Start"
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        if st.button(button_label):
            k=0
            button_label = "Stop"
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame = video_idetification(frame)
                # Display the resulting frame in a popup window
                # cv2.imshow('Camera Feed', frame)
                stframe.image(frame, channels="BGR")
                if k==0:
                    k=1
                    if st.button(button_label):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                
            # Release the camera and close the popup window
        
def tain_images():
    st.header('New Image uploader')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    user_input = st.text_input("Enter some text", "Default")
    if st.button("Generate") :
        temp_file_path = os.path.join(image_directory, uploaded_image.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        training_images_finc(temp_file_path, user_input)
        vector_store.persist()
        st.write("Face model generated")
def main():
    with tab1:
        tain_images()
    with tab2:
        st.header('Webcam')
        webcam_func()

if __name__ == "__main__":
     
     main()
