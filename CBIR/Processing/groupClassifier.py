import os
import torch
from torchvision import models, transforms

root = os.path.dirname(os.path.realpath(__file__))
WEIGHT_DIR = f"{root}/neural_weights/"


class groupClassifier():
    def __init__(self):

        # Check gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load trained model info
        best_model = torch.load(os.path.join(WEIGHT_DIR, 'groupClassWeights.pth.tar'),
        	map_location=self.device)
        self.mapping = best_model['class_mapping']


        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create model and load weights
        model = models.mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features,
            len(best_model['class_mapping'].values()))

        model.load_state_dict(best_model['state_dict'])
        self.model = model.eval()

        print("Pre-trained MOBNET group classifier neural model loaded.")


    def _predict_content(self, image=None):
        with torch.no_grad():

            # Apply image transformation
            input = self.transform(image).float()

            # CPU or GPU calculation
            input.to(self.device)
            
            input = input.unsqueeze(0)

            # Get model outputs and features 
            output = self.model(input)

            probs = torch.nn.functional.softmax(output[0])

            #print(probs)
            #image.show()
            
            res = torch.topk(probs,1)
            if res[0] < 0.7:
                return -1
            else:
                return res[1].item()