import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import shap
import pandas as pd
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F

# from models import Model_Rational_Label

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

model_path = "uw-hai/polyjuice"
poly_generator = pipeline("text-generation", 
    model=AutoModelForCausalLM.from_pretrained(model_path), 
    tokenizer=AutoTokenizer.from_pretrained(model_path))

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objs as go

from base64 import b64encode

from flask_cors import CORS, cross_origin

app = Dash(__name__)
flask_app = app.server
cors = CORS(flask_app)
flask_app.config['CORS_HEADERS'] = 'Content-Type'

def graphProbs(probs, labels):
    label2color = {'hate speech':'indianred', 'normal':'darkgreen', 'offensive':'gold'}
    colors = [label2color[label] for label in labels]
    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=probs,
                                 hole=0.6,
                                )]
                    )
    fig.update_traces(
        title="Classification Probabilities",
        hoverinfo='label+percent',
        textinfo='label+percent',
        textfont_size=20,
        marker=dict(colors=colors, line=dict(color='DarkSlateGrey', width=2)))
    return fig

app.layout = html.Div(children=[
    html.H1(children='xHate Explainable Abusive Language Interface'),

    html.Div([
        "Input: ",
        dcc.Input(id='textbox-input', value="A dog is embraced by the woman.", type='text'),
    ]),
    html.Div(id='textbox-output',
             children=[
                 dcc.Graph(id='probs_graph', figure={
                    'data': [graphProbs(probs=[0,1,0], labels=['hate speech', 'normal', 'offensive'])],
                    'layout': go.Layout(title="Classification Probabilities")
                }),
                html.H1(children='SHAP_graph'),
                dcc.Graph(id='SHAP_graph', figure=px.bar()),
                html.H1(children='LIME-Dashboard'),
                html.P(id="LIME-Dashboard")
             ]),
    html.Div(id = 'polyjuice-output', 
             children = [html.H1(children='Select Polyjuice Control Code'),
                    dcc.Dropdown(['negation', 'quantifier', 'shuffle', 'lexical', 'resemantic', 'insert', 'delete', 'restructure'], 'negation', id='demo-dropdown'),
                    html.Div(id='dd-output-container'),
                    dcc.Graph(id='probs_graph_poly', figure={
                    'data': [graphProbs(probs=[0,1,0], labels=['hate speech', 'normal', 'offensive'])],
                    'layout': go.Layout(title="Classification Probabilities")
                            })],
                    ),
])
    

def predict(text):
    text = str(text)
    inputs = tokenizer(str(text), return_tensors="pt")
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)\
            .detach().numpy().astype(float)[0].round(decimals=2)
    return probs


@app.callback(
    Output(component_id='probs_graph', component_property='figure'),
    Input(component_id='textbox-input', component_property='value')
)
def updateProbsGraph(input_value):
    res = _predict(input_value)
    labels, probs = res['labels'], res['probs']

    if input_value.isspace() or input_value == "" or input_value == "What's happening?":
        probs = [int(label == 'normal') for label in labels]

    probs_graph = graphProbs(probs, labels)
    probs_graph.update_layout(transition_duration=500)

    return probs_graph

pred = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

def _predict(text):
    [results] = pred(text)
    labels = [result['label'] for result in results]
    probs = [result['score'] for result in results]

    return {'labels': labels, 'probs': probs}

explainer = shap.Explainer(pred)
@app.callback(
    Output(component_id='SHAP_graph', component_property='figure'),
    Input(component_id='textbox-input', component_property='value')
)
def updateSHAP(input_value):
    res = _predict(input_value)
    labels, probs = res['labels'], res['probs']
    label2color = {'hate speech':'indianred', 'normal':'darkgreen', 'offensive':'gold'}
    # colors = [label2color[label] for label in labels]

    if input_value.isspace() or input_value == "" or input_value == "What's happening?":
        probs = [int(label == 'normal') for label in labels]

    shap_values = explainer([input_value])
    temp_len = len(shap_values.data[0])

    toks = list(shap_values[0,1:temp_len-1,:].data)
    data = {"Token": [], 'value': [], 'label': []}

    for i, label in enumerate(labels):
        data['Token'] += toks
        data['value'] += list(shap_values[0,1:temp_len-1,i].values)
        data['label'] += [label]*len(toks)

    df = pd.DataFrame(data=data)

    SHAP_graph = px.bar(df, x='Token', y='value',
                        color='label', color_discrete_map=label2color,
                        barmode='group')
    return SHAP_graph

@app.callback(
    Output(component_id="LIME-Dashboard", component_property='children'),
    Input(component_id='textbox-input', component_property='value')
)
def update_LIME(text):
    explainer = LimeTextExplainer(class_names=['Hate-Speech', 'Normal', 'Offensive'])

    exp = explainer.explain_instance(text, predict_prob)

    obj = html.Iframe(
            # Javascript is disabled from running in an Iframe for security reasons
            # Static HTML only!!!
            srcDoc=exp.as_html(),
            width='100%',
            height='150px',
            style={'border': '2px #d3d3d3 solid'},
        )
    return obj

def predict_prob(text):
    outputs = model(**tokenizer(text, return_tensors="pt", padding=True))
    probas = F.softmax(outputs.logits).detach().numpy()
    return probas


@app.callback(
    Output('dd-output-container', 'children'),
    Input('textbox-input', 'value'),
    Input('demo-dropdown', 'value'),
)

def generate_polyjuice(text, control_code):
    if not control_code:
        return
    prompt_text = text + " [" + control_code + "]"
    l = poly_generator(prompt_text, num_return_sequences=1)
    print(l)
    res = ""
    print(l[0]['generated_text'])
    if len(l[0]['generated_text'].split(" [" + control_code + "] ")) == 2:
        perturbed = l[0]['generated_text'].split(" [" + control_code + "] ")[1]
        if len(perturbed.split(" [SEP] ")) < 2:
            back = perturbed.split(" [SEP] ")
            back_split = back.split(" [ANSWER] ")
            res = back_split
        else:
            [front, back] = perturbed.split(" [SEP] ")
            front_split = front.split("[BLANK]")
            back_split = back.split(" [ANSWER] ")
            print(front, back)
            if len(front_split) == len(back_split):
                for i in range(len(front_split)):
                    res += front_split[i]
                    res += back_split[i]
    
    return res

@app.callback(
    Output(component_id='probs_graph_poly', component_property='figure'),
    Input(component_id='textbox-input', component_property='value'),
    Input('demo-dropdown', 'value'),
)
def updateProbsGraphPoly(text, control_code):
    input_value = generate_polyjuice(text, control_code)
    res = _predict(input_value)
    labels, probs = res['labels'], res['probs']

    if input_value.isspace() or input_value == "" or control_code == '':
        probs = [int(label == 'normal') for label in labels]

    probs_graph = graphProbs(probs, labels)
    probs_graph.update_layout(transition_duration=500)

    return probs_graph


@flask_app.route("/test")
@cross_origin()
def getExplainations():
    print("/test", flush=True)

    return 0

if __name__ == '__main__':
    print("main", flush=True)
    app.run_server(host="0.0.0.0", debug=True)