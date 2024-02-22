# prepare the emotion dataset
from datasets import load_dataset

emotions = load_dataset("emotion")
emotions.set_format(type="pandas")

def label_int2str(x):
    return emotions["train"].features["label"].int2str(x)

df = emotions["train"][:]
df["label_name"] = df["label"].apply(label_int2str)
print(df.head())
emotions.reset_format()

# tokenizer
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print("tokenizer's vocab size: ", tokenizer.vocab_size)
print("tokenizer's max length: ", tokenizer.model_max_length)

text = "Tokenizing text is a core task of NLP."
print("text:")
print(text)

encoded_text = tokenizer(text)
print("encoded_text: ")
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print("tokens:")
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))


def tokenize(batch):
    # a processing function to tokenize
    return tokenizer(batch["text"], padding=True, truncation=True)
print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)

# creating a text classifier
import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# extracting the last hidden states
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
print(f"Output tenser shape: {outputs.last_hidden_state.size()}")

def extract_hidden_states(batch):
    # place model inputs on the GPU
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    # extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# preparing training and validation datasets
import numpy as np

x_train = np.array(emotions_hidden["train"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
x_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_valid = np.array(emotions_hidden["validation"]["label"])
print("train shape: ", x_train.shape)
print("valid shape: ", x_valid.shape)

# creating a feature matrix
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

x_scaled = MinMaxScaler().fit_transform(x_train)
mapper = UMAP(n_components=2, metric="cosine").fit(x_scaled)
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
print(df_emb.head())

fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# training a simple classifier
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(x_train, y_train)
print("lr model of sklearn's score: ", lr_clf.score(x_valid, y_valid))

# fine-tuning transformers
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error"
)

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)

y_preds = np.argmax(preds_output.predictions, axis=1)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

plot_confusion_matrix(y_preds, y_valid, labels)

# error analysis
from torch.nn.functional import cross_entropy

def forward_pass_with_label(batch):
    # place all input tensors on the same device as the model
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = model(**inputs)
        pred_labels = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_labels.cpu().numpy()}

# convert out dataset back to PyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, batched=True, batch_size=16)

emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))

print(df_test.sort_values("loss", ascending=False).head(10))
print(df_test.sort_values("loss", ascending=True).head(10))

model.save_pretrained("./pretrained_model")
tokenizer.save_pretrained("./pretrained_model")