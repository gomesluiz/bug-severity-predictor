# Standard packages.
import io
import os
import xml.etree.ElementTree as ET

# Third-party packages.
import numpy as np
import requests
import torch
import transformers as ppb
import xgboost as xgb

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

# Setup application.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'wP4xQ8hU1jJ5oI1c'
bootstrap = Bootstrap(app)

# Define custom form.
class InputForm(FlaskForm):
    bug_id = StringField('Bug Id:', validators=[DataRequired()])


@app.route('/', methods=['GET', 'POST'])
def index():
    form        = InputForm(request.form)
    severity    = ''
    description = ''
    if form.validate_on_submit():
        description = read_bug_description(form.bug_id.data) 
        # TODO: clean description
        X = extract_features(description)
        severity = make_prediction(X) 

    return render_template('index.html', form=form, description=description, severity=str(severity))

def read_bug_description(id):
    url = f"https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id={id}"
    print(url)
    r = requests.get(url)
    if r.status_code == 404:
        return 'bug report not found!'

    f = io.StringIO(r.text)
    tree = ET.parse(f)
    root = tree.getroot()
    description = root.findall('./bug/long_desc/thetext')
    if description is None:
        return 'bug report without description'


    return description[0].text

def make_prediction(X):
    filename = os.path.join('data', 'model', 'final-model.bin')
    #dt = xgb.DMatrix(X)
    clf = xgb.XGBClassifier()
    bst = xgb.Booster({'nthread':4})
    bst.load_model(filename)
    clf._Booster = bst
    return clf.predict(X)[0]

def extract_features(description):
    max_len=128
    # Create DistilBERT model and tokenizer 
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, 
        ppb.DistilBertTokenizer, 
        'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model     = model_class.from_pretrained(pretrained_weights)
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
    

