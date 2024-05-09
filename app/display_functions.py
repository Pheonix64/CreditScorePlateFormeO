import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

color_code = 'rgba(0,0,0,0)'

def display_feat_import(loaded_model):
    # Extraire les importances des caractéristiques
    feature_importances = loaded_model.feature_importances_

    # Définir les noms et les explications des caractéristiques
    feature_names_fr = {
        'Monthly_Inhand_Salary': 'Représente le salaire net mensuel d\'une personne',
        'Num_Credit_Card': 'Représente le nombre d\'autres cartes de crédit détenues par la personne',
        'Interest_Rate': 'Représente le taux d\'intérêt sur la carte de crédit',
        'Delay_from_due_date': 'Représente le nombre moyen de jours de retard à partir de la date d\'échéance',
        'Num_of_Delayed_Payment': 'Représente le nombre moyen de paiements retardés par une personne',
        'Num_Credit_Inquiries': 'Représente le nombre de demandes de carte de crédit',
        'Outstanding_Debt': 'Représente le montant de la dette restante à payer',
        'Credit_History_Age': 'Représente l\'ancienneté de l\'historique de crédit de la personne',
        'Monthly_Balance': 'Représente le solde mensuel du client',
        'Credit_Mix_Encoded': 'Représente la classification du mix de crédits'
    }

    # Obtenir les noms des caractéristiques
    feature_names = list(feature_names_fr.keys())

    # Créer un DataFrame avec les noms des caractéristiques et leurs importances
    df = pd.DataFrame({'Caracteristiques': feature_names, 'Importance': feature_importances})

    # Trier le DataFrame par ordre d'importance décroissante
    df = df.sort_values(by='Importance', ascending=True)

    # Arrondir les valeurs d'importance à 3 chiffres après la virgule
    df['Importance'] = df['Importance'].round(3)

    # Créer un graphique à barres horizontales avec Plotly Express
    fig = px.bar(df, x='Importance', y='Caracteristiques', orientation='h', 
                color='Importance', color_continuous_scale='greens',
                labels={'Importance': 'Importance'},
                hover_data={'Caracteristiques': True, 'Importance': True},
                title='Importance des caractéristiques dans la prédiction')

    # Modifier le thème de mise en page pour changer l'arrière-plan et la taille de la figure
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Caractéristiques',
        xaxis=dict(tickformat=".3f"),  # Afficher 3 chiffres après la virgule
        bargap=0.2,
        bargroupgap=0.1,
        plot_bgcolor=color_code,  # Définir la couleur de l'arrière-plan
        paper_bgcolor=color_code,  # Définir la couleur du papier de l'arrière-plan
        title_x=0.5,  # Centrer le titre
        width=800,  # Définir la largeur de la figure
        height=500  # Définir la hauteur de la figure
    )

    # Ajouter les explications contextuelles lorsque le curseur passe sur les barres du graphique
    fig.update_traces(hovertemplate='<b>%{y}</b>: %{x}<br>%{customdata}', 
                    customdata=[feature_names_fr[name] for name in df['Caracteristiques']])
    
    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def display_model_metrics():
    # Création d'une palette de couleurs vertes peu denses
    green_palette = ['#52bf90', '#49ab81', '#419873'] 

    # Création d'un exemple de DataFrame avec les informations
    data = {
        'Metrique': ['Précision', 'Rappel', 'Exactitude'],
        'Valeur': [92 , 92, 92],  # Les valeurs sont en pourcentage
        'Description': [
            'Nombre de vrais positifs divisé par le nombre total de prédictions positives',
            'Nombre de vrais positifs divisé par le nombre total d\'instances pertinentes',
            'Taux de prédictions correctes'
        ]
    }

    # Création du DataFrame
    df_metrics = pd.DataFrame(data)

    # Création du graphique à barres verticales avec Plotly Express
    fig = px.bar(df_metrics, x='Metrique', y='Valeur', orientation='v', 
                color='Metrique', color_discrete_sequence=green_palette,
                labels={'Valeur': 'Valeur'},
                hover_data={'Metrique': True, 'Valeur': True, 'Description': True},
                title='Métriques de performance du modèle')

    # Mise en forme du graphique
    fig.update_layout(
        xaxis_title='Métrique',
        yaxis_title='Valeur (%)',  # Mettre à jour le titre de l'axe y
        bargap=0.2,
        bargroupgap=0.1,
        plot_bgcolor=color_code,  # Couleur de l'arrière-plan
        paper_bgcolor=color_code,  # Couleur du papier de l'arrière-plan
        title_x=0.5,  # Centrer le titre
        width=720,  # Largeur de la figure
        height=400,  # Hauteur de la figure
        yaxis=dict(tickformat="""\%%"""),  # Formater l'axe y en pourcentage
    )

    # Personnalisation du modèle hovertemplate pour afficher les descriptions des métriques
    fig.update_traces(hovertemplate='<b>%{x}</b>: %{y}%<br>%{customdata}', customdata=df_metrics['Description'], width=0.5)  


	# Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def create_credit_score_chart(predicted):
    # Fonction pour obtenir les bornes inférieure et supérieure du score en fonction de la prédiction
    def get_score_range(predicted):
        if predicted == 0:
            return (300, 579)
        elif predicted == 1:
            return (580, 669)
        else:
            return (670, 739)

    # Obtenir les bornes inférieure et supérieure du score
    score_inf, score_sup = get_score_range(predicted)

    # Fonction pour obtenir la couleur en fonction de la prédiction
    def get_color(predicted):
        if predicted == 0:
            return 'red'
        elif predicted == 1:
            return 'orange'
        else:
            return '#419873'

    # Créer une figure avec Plotly
    fig = go.Figure()

    # Ajouter l'indicateur de jauge pour la borne inférieure du score
    fig.add_trace(go.Indicator(
        value=score_inf,
        mode="number+delta+gauge",
        gauge={
            'axis': {'range': [None, 739], 'visible': False},
            'bar': {'color': get_color(predicted)}  # Utiliser la fonction get_color() pour obtenir la couleur de la barre
        },
        domain={'row': 0, 'column': 0}
    ))

    # Ajouter l'indicateur de nombre et de delta pour la borne supérieure du score
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=score_sup,
        delta={'reference': 300},
        domain={'row': 0, 'column': 1}
    ))

    # Mise à jour de la mise en page avec un titre général
    fig.update_layout(
        grid={'rows': 1, 'columns': 2},
        title={'text': "Intervalle du Score de Crédit du Client"},
        template={'data': {'indicator': [{
            'mode': "number+delta+gauge",
            'delta': {'reference': 300}
        }]}},
        title_x=0.5,
        bargroupgap=0.1,
        plot_bgcolor=color_code,  # Couleur de l'arrière-plan
        paper_bgcolor=color_code,  # Couleur du papier de l'arrière-plan
    )

    return fig