# from django.shortcuts import render
# from django.http import HttpResponse
# from .utils import add_numbers
# from .forms import ConstructionProgressForm
# from   .data.label_encode    import 


# def home(request):
#     return render(request, 'base.html')

# def show_addition_form(request):
#     """
#     Display a simple form with two input fields and a button to add numbers.
#     """
#     result = None  # Initialize result variable
#     if request.method == 'POST':
#         # Get the numbers from the form input
#         number1 = int(request.POST.get('number1', 0))
#         number2 = int(request.POST.get('number2', 0))
#         # Call the addition function
#         result = add_numbers(number1, number2)

#     # Render the template and pass the result
#     return render(request, 'addition.html', {'result': result})

# def check_progress(request):
#     if request.method == 'POST':
#         form = ConstructionProgressForm(request.POST, request.FILES)
#         if form.is_valid():
#             # Process the form data and handle the image and activity selection
#             image = form.cleaned_data['image']
            # activity = form.cleaned_data['activity']

            
#             # Example processing using add_numbers (assuming you need some result from this)
#             # For this example, I'm going to call it with arbitrary numbers
#             # You may need to adjust this based on actual use
#             # e.g., numbers might come from some other form fields or calculations
#             # result = add_numbers(1, 2)  # Example usage
            
#             return render(request, 'result.html', {
#                 'activity': activity,
#                 'image': image
#             })
#     else:
#         form = ConstructionProgressForm()
    
#     return render(request, 'check_progress.html', {'form': form})

# import torch
# import joblib
# import numpy as np
# from PIL import Image
# from django.shortcuts import render
# from django.conf import settings
# from .forms import UploadImageForm
# from torchvision import transforms
# from torch.utils.data import Dataset
# from torchvision import models
from django.core.files.storage import FileSystemStorage

# # Load the model and label encoder
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #File path
# model_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\resnet_model.pth'
# label_encoder_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\label_encoder.pkl'

# resnet_model = models.resnet18(pretrained=False)
# num_features = resnet_model.fc.in_features
# label_encoder = joblib.load(label_encoder_path)
# resnet_model.fc = torch.nn.Linear(num_features, len(label_encoder.classes_))
# resnet_model.load_state_dict(torch.load(model_path))
# resnet_model.to(device)
# resnet_model.eval()

# # Define the transformations for data
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def predict_stage(image):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = resnet_model(image)
#         predicted_probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
#         predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probabilities)])[0] 

#     return predicted_label

# def check_progress(request):
#     if request.method == 'POST':
#         form = UploadImageForm(request.POST, request.FILES)
#         if form.is_valid():
#             image = form.cleaned_data['image'] 

#             # Save the image using Django's file storage system
#             fs = FileSystemStorage()                            
#             filename = fs.save(image.name, image) 
#             image_url = fs.url(filename)  # Get the URL for the saved image

#             # Convert and predict the construction stage
#             image = Image.open(image).convert('RGB')
#             predicted_stage = predict_stage(image) 
#             # image = form.cleaned_data['image'] 
#             # image = Image.open(image).convert('RGB')
#             # predicted_stage = predict_stage(image) 
#             return render(request, 'result.html', {'form': form, 'image':image,'predicted_stage': predicted_stage}) 
#     else:
#         form = UploadImageForm()

#     return render(request, 'check_progress.html', {'form': form})

import torch
import joblib
import numpy as np
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from .forms import UploadImageForm
from torchvision import transforms
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def home(request):
    return render(request, 'base.html')
# def upload(request):
#     return render(request, 'check_progress.html')
# Load the model, label encoder, and GPT-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update the file paths to the absolute path
# model_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\resnet_model.pth'
# label_encoder_path = 'C:\\Users\\Asus\\Downloads\\djangodemo\\djangodemo\\calc\\data\\label_encoder.pkl'
model_path = os.path.join(settings.BASE_DIR, 'calc', 'data', 'resnet_model.pth')
label_encoder_path = os.path.join(settings.BASE_DIR, 'calc', 'data', 'label_encoder.pkl')

resnet_model = models.resnet18(pretrained=False)
num_features = resnet_model.fc.in_features
label_encoder = joblib.load(label_encoder_path)
resnet_model.fc = torch.nn.Linear(num_features, len(label_encoder.classes_))
resnet_model.load_state_dict(torch.load(model_path))
resnet_model.to(device)
resnet_model.eval()


# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the transformations for data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_text(prompt, num_sequences=1, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    outputs = gpt2_model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=num_sequences, 
        do_sample=True, 
        top_p=0.95, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def predict_stage(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = resnet_model(image)
        predicted_probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probabilities)])[0]

    return predicted_label

def check_progress(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            # activity = form.cleaned_data['activity']

            image = Image.open(image).convert('RGB')
            predicted_stage = predict_stage(image)


            image = form.cleaned_data['image'] 
        #     Save the image using Django's file storage system
            fs = FileSystemStorage()                            
            filename = fs.save(image.name, image) 
            image_url = fs.url(filename)  # Get the URL for the saved image

            # Generate descriptive text based on the predicted label
            prompts = {
                'Foundation': "The foundation stage of construction involves laying the base of a building, including excavation and concrete pouring.",
                'Super-structure': "The super-structure stage includes the construction of the main frame of the building, including walls, floors, and roof.",
                'Interiors': "The interiors stage focuses on the internal finishing of the building, such as drywall, flooring, and fixtures."
            }
            prompt = prompts.get(predicted_stage, "The construction stage is not clearly defined.")
            generated_text = generate_text(prompt, num_sequences=1)

            return render(request, 'result.html', {'form': form,'image_url':image_url, 'predicted_stage': predicted_stage, 'generated_text': generated_text})
    else:
        form = UploadImageForm()

    return render(request, 'check_progress.html', {'form': form}) 

