from flask import Flask, jsonify, request, render_template,redirect,url_for
import time, json
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from display_functions import display_feat_import, display_model_metrics,create_credit_score_chart

app = Flask(__name__)


filename = 'model/credit_op_score_model.sav'

def load_model():
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model




#Load the model
loaded_model = load_model()
list_data = []

# Route GET pour récupérer des données
@app.route('/', methods=['GET'])
def get_data():
    return {"Message": "Bienvenue sur la plateforme!"}



@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données d'entrée
    data = request.get_json(force=True)

    # Vérifier si toutes les clés nécessaires sont présentes dans les données
    required_keys = ['monthly_inhand_salary', 'num_credit_card', 'interest_rate', 'delay_from_due_date', 
                     'num_of_delayed_payment', 'num_credit_inquiries', 'outstanding_debt', 'credit_history_age', 
                     'monthly_balance', 'credit_mix_encoded']
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'La clé "{key}" est manquante dans les données.'}), 400

    # Exemple de récupération de paramètres
    monthly_inhand_salary = data['monthly_inhand_salary']
    num_credit_card = data['num_credit_card']
    interest_rate = data['interest_rate']
    delay_from_due_date = data['delay_from_due_date']
    num_of_delayed_payment = data['num_of_delayed_payment']
    num_credit_inquiries = data['num_credit_inquiries']
    outstanding_debt = data['outstanding_debt']
    credit_history_age = data['credit_history_age']
    monthly_balance = data['monthly_balance']
    credit_mix_encoded = data['credit_mix_encoded']

    # Convertir les données en tableau numpy
    features = np.array([
        [monthly_inhand_salary, num_credit_card, interest_rate, delay_from_due_date,
         num_of_delayed_payment, num_credit_inquiries, outstanding_debt, credit_history_age,
         monthly_balance, credit_mix_encoded]
    ])

    # Faire la prédiction
    prediction = loaded_model.predict(features)

    client_status = ""

    if int(prediction[0]) == 0:
        client_status ="""Le score de crédit du client est bas, ce qui indique un risque élevé.
        Il est recommandé de ne pas approuver sa demande."""
    elif int(prediction[0]) == 1:
        client_status ="""Le score de crédit du client est moyen, suggérant un risque modéré.
        Vous pouvez envisager d'approuver sa demande sous certaines conditions."""
    else:
        client_status ="""Le score de crédit du client est élevé, ce qui témoigne de sa fiabilité.
        Vous pouvez approuver sa demande en toute confiance."""     
        
        
    #Figure pour afficher les caracteristiques les plus importances
    fig1 = display_feat_import(loaded_model)
    
    #Figure pour afficher les métriques du modèle
    fig2 = display_model_metrics()
    
    #Afficher l'intervalle de score de crédit du client
    fig3 = create_credit_score_chart(int(prediction[0]))
    
    list_data = [fig1, fig2, fig3, client_status]
    
    # Retourner la prédiction
    return  render_template('resultats.html', outputData=list_data)



if __name__ == '__main__':
    app.run(debug=True)
