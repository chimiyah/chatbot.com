import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
        
#DISCOUNT
#"The list below is the following scholarship programs: \n\nGraduating Classes \nwith highest honors (100% discount), \nwith high honors(50% discount), \nwith honors (25% discount), \n\nDean's List \nGWA of 1.25 or better (100% discount), \nGWA of 1.50 or better (50% discount), \nNote: students must not obtain grades below 88% or 2.00 at any subject \n\nPeforming Arts Group \nresidency as a member of at least 2.5 years will get 25% discount \n\nSchool Paper Staff \nEditor-in-chief(100% discount), \nAssociate Editor(50% discount) \nManaging Editor(25% discount) \nChief Photographer(25% discoount) \n\nSupreme Student Council \nPresident(100% discount) \nVice President(100%) \nSecrerary, treasurer, Auditor, Business Manager and PRO (25% discount) \n\nCollege Student Council(15% discount) \n-for President only "

#ADMISSION
#"For Freshmen \n -Birth Certificate (PSA Issued) \n -Form 138 (Original Report Card) \n-Good Moral Character Certificate\n -National Career Assessment Examination Result or Occupational Interest Inventory for HS Cert of Rating (For TESDA Programs only) \n -3 pcs. 2×2 Colored Picture \n -1 Long White Folder\n -1 Long Brown Envelope\n\nFor Transferees/Degree Holders\n -Birth Certificate (PSA Issued)\n -Transfer Credential\n\nGood Moral Character Certificate\n -Transcript of Records/ Certified True Copy of Grades\n -3 pcs. 2×2 Colored Picture\n -1 Long White Folder\n -1 Long Brown Envelope \n\nFor Cross Enrollees\n -Permit to Cross-Enroll Certificate\n -Birth Certificate (PSA Issued)\n -2 pcs. 2×2 Colored Pictures\n -1 Long White Folder\n -1 Long Brown Envelope"

#SCHOLAR
#"The list below is the following scholarship programs: \n\nGraduating Classes \nwith highest honors (100% discount), \nwith high honors(50% discount), \nwith honors (25% discount) \n\nDean's List \nPeforming Arts Group \nALTAS Angels \nThe Perpetualite / School Paper Staff \nSupreme Student Council \nCollege Student Council \nSpecial Scholarships \n"
