import os

import joblib
import numpy as np
import torch
import transformers as ppb
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'wP4xQ8hU1jJ5oI1c'
bootstrap = Bootstrap(app)

class InputForm(FlaskForm):
    bug_id = FloatField('Bug Id:', validators=[DataRequired()])

# 
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, 
    ppb.DistilBertTokenizer, 
    'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model     = model_class.from_pretrained(pretrained_weights)

@app.route('/', methods=['GET', 'POST'])
def index():
    form     = InputForm(request.form)
    severity = 'No-image'
    if form.validate_on_submit():
        id = [[form.bug_id.data]]
        description = "Email reminders are added by default to new events - great feature.  However, editing an event in Lightning will remove this email reminder.  I have been unable to find a way to add the reminder back after it's been lost."
        X = extract_features(description)
        severity = make_prediction(X) 
        print(severity)

    return render_template('index.html', form=form, severity=severity)

def make_prediction(X):
    filename = os.path.join('data', 'model', 'final-model.joblib')
    model = joblib.load(filename)
    return model.predict(X)[0]

def extract_features(description):
    max_len=64
    sentence = ' '.join(description.split()[:max_len])
    tokenized = tokenizer.encode(sentence, add_special_tokens=True) 
    
    padded = np.array([tokenized + [0]*(max_len-len(tokenized))])
    
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    return features
    

