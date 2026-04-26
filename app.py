import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import base64

#explaining the outputted prediction 
def explain_prediction(label):
    explanations = {
        "AAA": "High uniformity, smooth texture, and minimal defects indicate top-grade beans.",
        "AA": "Good consistency and shape with only minor imperfections.",
        "A": "Moderate quality with some visible variation in size or texture.",
        "AB": "Mixed characteristics, showing both high and lower quality features.",
        "PB-I": "Peaberry beans with relatively uniform shape and size.",
        "PB-II": "Peaberry beans with more variation and slight imperfections.",
        "Bulk": "Lower uniformity and mixed bean quality.",
        "Bits": "Broken or fragmented beans indicating low quality.",
        "C": "Irregular shapes and defects suggest lower-grade beans."
    }
    
    return explanations.get(label, "Prediction based on learned visual patterns in the dataset.")

#converting image to text (embeds into the app's styling - no need for separate file path)
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("beans.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #f5e6d3;
        background-image: 
            url("data:image/png;base64,{img}"),
            url("data:image/png;base64,{img}");
        background-repeat: repeat-y, repeat-y;
        background-position: left top, right top;
        background-size: 120px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


#loading resnet18 (neural net trained on image net that can recognize patterns in images)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]) #removing last layer because original resnet does image --> features --> final classification. we are stopping one part before with images --> features
model.eval()

#transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#feature function
def get_features(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features.flatten().numpy().reshape(1, -1)

#loading trained classifier: allows model to go through each folder, read the images, turn them into features and then store them with labels
from sklearn.linear_model import LogisticRegression

X = []
y = []

data_dir = "computer-vision-project-main/CBD_Coffee Bean Dataset"

for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    
    if os.path.isdir(folder_path):
        for i, file in enumerate(os.listdir(folder_path)):
            if i > 30:
                break
            
            path = os.path.join(folder_path, file)
            try:
                img = Image.open(path).convert("RGB")
                feats = get_features(img)
                X.append(feats.flatten())
                y.append(label)
            except:
                pass

X = np.array(X)
y = np.array(y)

#turning numbers into labels with logistic regression (after resnet turned images into numbers)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

#streamlit
st.title("☕ Coffee Bean Classifier")

st.write("Upload a coffee bean image and the model will predict its quality grade.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", width=500)

    features = get_features(image)
    prediction = clf.predict(features)

    st.write("### Prediction:")
    st.success(prediction[0])

    st.write("### Reasoning:")
    st.info(explain_prediction(prediction[0]))
