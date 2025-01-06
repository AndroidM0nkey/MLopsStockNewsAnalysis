import torch
from sklearn.preprocessing import LabelEncoder
from train import SentimentClassifier
from transformers import AutoTokenizer

model_path = "sentiment_model.pth"
model_name = "bert-base-uncased"


def prepare_model():
    model = SentimentClassifier(model_name=model_name, num_classes=3)
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )

    model.eval()

    return model


def predict_sentiment(model):
    text = input("Enter the text for sentiment analysis: ")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(
        text, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predicted_class = logits.argmax(dim=-1).item()

    label_encoder = LabelEncoder()
    sentiment_label = label_encoder.inverse_transform([predicted_class])[0]
    return sentiment_label
