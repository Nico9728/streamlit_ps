import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


atp = pd.read_csv("atp_data.csv", sep=";")
#sidebar
st.sidebar.title("Sommaire")
#menu pour naviguer
pages= ["Présentation du projet", "Exploration", "DataVizualisation","Préparation des données","Machine Learning","Conclusion"]
page=st.sidebar.radio("Selectionnez une partie",pages)

#élément linkedin dans le sidebar
st.sidebar.markdown("---")  
st.sidebar.markdown("### Membres du projet")
icon_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
st.sidebar.markdown(f'<a href="https://www.linkedin.com/in/laura-daulasim-03840a204/"><img src="{icon_url}" width="15" style="vertical-align: middle;"> Laura Daulasim</a>', unsafe_allow_html=True)
st.sidebar.markdown(f'<a href="https://www.linkedin.com/in/laurent-grondin-337a48156/"><img src="{icon_url}" width="15" style="vertical-align: middle;"> Laurent Grondin</a>', unsafe_allow_html=True)
st.sidebar.markdown(f'<a href="https://www.linkedin.com/in/patrick-singh-1a012510a/"><img src="{icon_url}" width="15" style="vertical-align: middle;"> Patrick Singh</a>', unsafe_allow_html=True)
st.sidebar.markdown(f'<a href="https://www.linkedin.com/in/nicolas-marthely-5212778b/"><img src="{icon_url}" width="15" style="vertical-align: middle;"> Marthely Nicolas</a>', unsafe_allow_html=True)


if page == pages[0] : 
  image = r"https://github.com/Nico9728/streamlit_ps/blob/main/tennis4.jpg"
  st.image(image, width=500)
  st.title("Projet Paris Sportif")
  st.header("Présentation du projet")
  st.write("""
  Dans le cadre de notre formation de Data Analyst, nous avons choisi d'exploiter le potentiel du machine learning pour prédire les résultats des matchs de tennis.
           
  **L'objectif de notre projet  est de développer un modèle prédictif capable d'estimer la probabilité de victoire d'un joueur.**
           
  Notre démarche s'appuie sur une analyse approfondie des données historiques des matchs de l'ATP (2000–2018) qui regroupent les performances individuelles des joueurs ainsi que la prise en compte des caractéristiques des surfaces de jeux ou encore le classement de ces derrniers.""")


if page == pages[1] : 
  st.header("Exploration des données du projet")
  st.write("### Le jeu de données")
  st.write("""
  Notre projet se base sur le DataFrame 'atp_data.csv' qui regroupe un certain nombre de données relatives à tous les matchs ATP entre les années 2000 et 2018.
           
  Pour commencer, nous avons veillé à comprendre chaque variable, ce qu’elle décrit, si elle est prévisible ou non, son type informatique, son taux de valeurs manquantes.
  """)
  
  st.write("Données du DataFrame:")
  st.dataframe(atp.head(10))
  st.write(atp.shape)
  st.write("Description des variables numériques du DataFrame:")
  st.dataframe(atp.describe())

  st.write("Format des données :")
  atp.dtypes

  if st.checkbox("Afficher les NA") :
      st.dataframe(atp.isna().sum())
  

  st.write("""
  Ces informations primaires sont très intéressantes car elles vont nous permettre de mieux comprendre les données que l’on va traiter, cibler celles qui seront effectivement nécessaires pour atteindre notre objectif mais également exclure celles dont la pertinence est moindre.
           
  Cela nous permet d’avoir une première visualisation de la qualité de nos données (notamment en cas de données manquantes ou de types de données non pertinents), afin de réfléchir à des solutions pour pallier ce problème et formuler des hypothèses sur leur traitement.""")


if page == pages[2] : 
  st.header("DataVizualisation")
  st.write("Dans cette section, nous présentons des représentations graphiques tirées de notre Data Frame, afin de visualiser et étudier les relations entre les différentes variables.")

  st.subheader("***Heatmap de correlations***")
  # Texte masqué avec st.expander
  with st.expander("Pourquoi la Heatmap ?"):
    st.write("La heatmap est un outil visuel qui simplifie la compréhension des relations entre variables, en mettant en évidence les corrélations positives (près de 1), négatives (près de -1) ou faibles (près de 0).")
    st.write("Elle est particulièrement utile pour détecter rapidement les tendances et résumer des données complexes en une vue accessible, même pour les non-spécialistes.")

               
  numerical_features = atp.select_dtypes(include=np.number).columns
  cor = atp[numerical_features].corr()
  fig, ax = plt.subplots(figsize=(14, 14))
  sns.heatmap(cor, annot=True, ax=ax, cmap="coolwarm")
  plt.show()
  st.write(fig)
  with st.expander("Interprétation du graphique"):
    st.write(
    "La heatmap de corrélation nous a permis de détecter des relations significatives entre différents indicateurs liés aux performances et aux prédictions de victoire. "
    "\n\nVoici un résumé des principales conclusions :"
    "\n\n- **Relation entre 'Best of' et 'WSets'** : Une corrélation positive forte (0,82) souligne qu'un format avec un maximum de sets gagnants permet au vainqueur d'en remporter davantage."
    "\n- **Cohérence des bookmakers** : Les cotes gagnantes (PSW et B365W, corrélation de 0,98) et perdantes (PSL et B365L, corrélation de 0,89) montrent une grande homogénéité dans leurs prédictions."
    "\n- **Proba_elo et elo_winner** : Une corrélation modérée (0,66) reflète une certaine compatibilité entre l'évaluation ELO du vainqueur et la probabilité associée."
    "\n- **Relations inverses importantes** : Des corrélations négatives significatives (-0,61 entre Proba_elo et PSW, et -0,66 avec B365W) indiquent une relation inverse logique entre la probabilité de victoire basée sur ELO et les cotes des bookmakers.")

  #GRAPHIQUE NUMERO 2
  st.subheader("***Diagrammes en barre***")
  with st.expander("Pourquoi les diagrammes en barre ?"):
    st.write("Les diagrammes en barre, grâce à leur simplicité, facilitent l'analyse et mettent en évidence des tendances intéressantes."
            " Ils permettent de comparer et d'interpreter facilement des données.")
  #conversion de la date en date time
  atp['Date']=pd.to_datetime(atp['Date'],format='%d/%m/%Y')

  # Compter les victoires par joueur
  victoires = atp['Winner'].value_counts()

  # Sélectionner les 10 joueurs avec le plus de victoires
  top_victoires = victoires.head(10)

  # Créer le diagramme en barres
  fig, ax = plt.subplots(figsize=(10, 6))
  top_victoires.plot(kind='bar', color='skyblue', edgecolor='black')
  plt.title('Top 10 joueurs avec le plus de victoires', fontsize=14)
  plt.xlabel('Joueur', fontsize=12)
  plt.ylabel('Nombre de victoires', fontsize=12)
  plt.xticks(rotation=45, ha='right', fontsize=10)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()
  # Afficher le diagramme
  st.pyplot(fig)


  #GRAPHIQUE NUMERO 3
  # Identification les 10 joueurs avec le plus de victoires au total
  top_10_players = atp['Winner'].value_counts().head(10).index

  # Filtrage les données pour les années 2015, 2016, 2017, 2018
  years = [2015, 2016, 2017, 2018]
  victoires_by_year = {year: atp[atp['Date'].dt.year == year]['Winner'].value_counts() for year in years}

  # Création une structure de données pour les victoires des top 10 joueurs par année
  victoires_data = {player: [victoires_by_year[year].get(player, 0) for year in years] for player in top_10_players}

  # Convertion les données en DataFrame pour une visualisation plus simple
  victoires_df = pd.DataFrame(victoires_data, index=years)

  # Création graphique en barres groupées par joueur
  x = np.arange(len(top_10_players))  # Positions des groupes sur l'axe X
  bar_width = 0.2  # Largeur des barres

  fig, ax = plt.subplots (figsize=(14, 7))
  for i, year in enumerate(years):
    plt.bar(x + i * bar_width, victoires_df.iloc[i], width=bar_width, label=str(year))

  # Ajout des légendes, titres et labels
  plt.title('Évolution des victoires des 10 meilleurs joueurs (2015, 2016, 2017, 2018)', fontsize=16)
  plt.xlabel('Joueur', fontsize=12)
  plt.ylabel('Nombre de victoires', fontsize=12)
  plt.xticks(x + bar_width * (len(years) - 1) / 2, top_10_players, rotation=45, ha='right', fontsize=10)  # Ajuster les labels des joueurs
  plt.legend(title='Année', fontsize=10)
  plt.tight_layout()
  # Affichage du graphique
  st.pyplot(fig)

  #synthèse 
  with st.expander("Interprétation des graphiques"):
    st.write ("Les graphiques révèlent que Roger Federer est le joueur avec le plus grand nombre de victoires globales parmi les meilleurs joueurs de tennis de notre data set. ")
    st.write("Cependant, l'évolution annuelle des victoires montre des fluctuations notables pour tous les joueurs, y compris Nadal et Djokovic, indiquant des variations de performances d'une année à l'autre. ")
    st.write("Cela souligne la dynamique compétitive même au sommet et le fait qu'avec les années, la domination se fait moins forte chez les top playeurs.")

  #GRAPHIQUE 4
  # Distribution des classements des gagnants vs perdants
  
  # Filtrage pour mieux voir le graph
  df_vis = atp[(atp.WRank < 150) & (atp.LRank < 150)]

  # Création du graphique
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.histplot(data=df_vis, x='WRank', bins=50, kde=True, color='blue', label='Classement des gagnants')
  sns.histplot(data=df_vis, x='LRank', bins=50, kde=True, color='red', label='Classement des perdants')
  plt.legend()
  plt.title("Distribution des classements des joueurs (gagnants vs perdants)")
  plt.xlabel("Classement")
  plt.ylabel("Fréquence")
  # Afficher le graphique
  st.pyplot(fig)

  #Synthèse
  with st.expander("Interprétation du graphique"):
    st.write("Ce diagramme de distribution nous a permis d'apprecier la répartition de la fréquence des matchs gagnés et perdus en fonction du classement. ")
    st.write("Il met en lumière que les joueurs mieux classés gagnent plus fréquemment, tandis que ceux avec un classement plus faible remportent moins de matchs.")
    st.write("Cependant, entre les rangs 35 et 40, les courbes se croisent, révélant un équilibre entre les victoires et les défaites à ce niveau.")
    st.write("Cela confirme que même les joueurs de haut rang peuvent perdre, tandis que ceux moins bien classés ont également des opportunités de succès.")



  #GRAPHIQUE 5
  #Relation entre la différence de élo et la probabilité prédite

  st.subheader("***Nuage de points***")
  with st.expander("Pourquoi le nuage de points ?"):
    st.write("Le nuage de points est utile en data analyse pour visualiser la relation entre deux variables et identifier des tendances,"
    " des clusters ou encore des anomalies.")
  
  # Calcul de la différence de Elo
  atp['elo_diff'] = atp['elo_winner'] - atp['elo_loser']

  # Créaion du graphique
  fig, ax = plt.subplots(figsize=(12, 6))  # Correction pour initialiser fig et ax
  scatter = ax.scatter(
    atp['elo_diff'], atp['proba_elo'], 
    c=atp['proba_elo'], cmap='coolwarm', alpha=0.7, edgecolor='k')
  # Ajout une barre de couleur pour donner une échelle
  cbar = plt.colorbar(scatter, ax=ax)
  cbar.set_label('Probabilité Elo')

  # Ajout des labels et un titre
  ax.set_xlabel('Différence de Elo')
  ax.set_ylabel('Probabilité basée sur Elo')
  ax.set_title('Relation entre la différence de Elo et la probabilité')

  # Afficher le graphique dans Streamlit
  st.pyplot(fig)

  #Synthèse
  with st.expander("Interprétation du graphique"):
    st.write("Notre nuage de points illustre la corrélation entre la différence de Elo (elo_winner - elo_loser) et la probabilité Elo (proba_elo).")
    st.write(" Il montre que plus le Elo du gagnant dépasse celui du perdant, plus la probabilité de victoire du gagnant est élevée, "
    "confirmant une relation attendue entre ces variables.")

  #Enseignement
  st.subheader("Principaux enseignements:")
  st.write("A travers nos visualisations, nous avons pu mettre en évidence des relations fortes, comme la corrélation entre les classements Elo et les probabilités de victoire,tout en confirmant que les joueurs mieux classés gagnent généralement davantage. ")
  st.write("Cependant, nous avons également pu constater la dynamique imprévisible du tennis, où des favoris peuvent être surpris par des outsiders et où les performances des meilleurs joueurs fluctuent avec le temps, reflétant l'incertitude inhérente à ce sport compétitif.")

 


if page == pages [3]:
  
  st.header("Préparation des donneés")
  st.write("Dans cette section, nous allons illustrer les étapes réalisées pour préparer nos données pour le Machine learning.")

 
  if st.checkbox("Afficher les infos du jeu de données:") :
    # Capturer la sortie de .info()
    import io
    buffer = io.StringIO()
    atp.info(buf=buffer)  
    info = buffer.getvalue()

    # Afficher l'information dans Streamlit
    st.write("**Informations sur le DataFrame**")
    st.write("Nous avons commencé par vérifier si le type de nos variables étaient dans un bon format pour les exploiter dans nos modèles.")
    st.text(info)
  
  # Correction format date
  st.subheader("1/ Correction du format des données")
  st.write("Un mauvais format de données pouvant affecter notre modélisation, nous avons converti la variable 'Date' au bon format.")

 
  with st.expander("Afficher les donées formatées"):
    # affichage du code dans streamlit 
       # création de la variable code pour l'afficher
      code= "atp['Date']=pd.to_datetime(atp['Date'],format='%d/%m/%Y')"
      st.code(code,language='python')
      atp['Date']=pd.to_datetime(atp['Date'],format='%d/%m/%Y')

      if st.checkbox("Afficher le format des données:") :
        st.dataframe(atp.dtypes)
  
  st.subheader("2/ Création de notre variable cible")
  st.write("Une fois le contrôle de nos variables réalisé, nous avons fait le choix de créer une nouvelle variable qui deviendra notre variable à prédir pour la suite de notre projet.")
  
  with st.expander("Afficher la nouvelle variable et ses proportions"):
      # création de la variable code pour l'afficher
      code= "atp['target'] = (atp['elo_winner'] > atp['elo_loser']).astype(int)"
      st.code(code,language='python')
      atp['target'] = (atp['elo_winner'] > atp['elo_loser']).astype(int)
  
      st.write("Cette nouvelle variable **target**  créee en comparant la probabilité de victoire des deux joueurs nous permet d'avoir une logique de Classification où nous pouvons déterminer si un match "
      "a été  remporté par un favori = "" 1 ""  ou un outsider = "" 0 "".")
 

      #tableau de proportion
      atp['target'].value_counts(normalize=True)*100

      #graphique
      fig, ax = plt.subplots()
      labels=["Favoris gagnants", "Outsiders gagnants"]
      ax.pie(atp['target'].value_counts(normalize=True)*100, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
      ax.axis('equal')  # pour s'assurer que le cercle est bien rond
      plt.title("Répartition des Victoires")
      # Affichage dans Streamlit
      st.pyplot(fig)
      #commentaire
      st.write("*La proportion de victoires des favoris est, de manière logique, plus élevée que celle des outsiders, car les favoris remportent la majorité de leurs matchs.*")
  
  st.subheader("3/ Choix des variables")
  st.write("Une fois notre variable cible créee, nous avons réduit la taille de notre jeu de données afin de garder uniquement les variables pertinantes.")

  with st.expander("Afficher le Data Frame réduit"):
    liste=['Comment','Date','Best of','Location','Tournament','PSW','PSL','B365W','B365L','Loser','Winner','ATP','elo_loser','elo_winner','proba_elo']
    atp=atp.drop(columns=liste,axis=1)
    st.dataframe(atp.head(10))
    st.write(atp.shape)
    st.write("Cette réduction vise à accroître les performances et l'efficacité des modèles tout en diminuant le temps de calcul et l'utilisation des ressources.")

  st.subheader("4/ Imputation, Standardisation et encodage des variables selectionnées.")
  st.write("La dernière étape de la préparation des données a impliqué:"
  "\n- l'imputation via le **SimpleImputer** pour remplacer les **valeurs manquantes**"
  "\n- la standardisation des variables numériques via le **StandardScaler pour assurer leur comparabilité**"
  "\n- ainsi que **l'encodage** des variables catégorielles à l'aide du **OneHotEncoder**")  
  st.write("Le respect de ces étapes garantira une cohérence optimale des données, essentielle pour améliorer la **précision, la robustesse et l'efficacité** des modèles de machines learning.")

  with st.expander("Afficher le Data Frame final"):
    from sklearn.model_selection import train_test_split

    #Séparation jeu test et entrainement
    X = atp.drop('target',axis=1)
    y = atp['target']

    # Diviser en jeux d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Séparer les colonnes numériques et catégorielles pour le jeu d'entraînement
    num_train = X_train.select_dtypes(include=['number'])
    cat_train = X_train.select_dtypes(include=['object'])
    # Séparer les colonnes numériques et catégorielles pour le jeu de test
    num_test = X_test.select_dtypes(include=['number'])
    cat_test = X_test.select_dtypes(include=['object'])

    #remplacement valeurs manquante avec le simple Imputer
    from sklearn.impute import SimpleImputer

    # Imputation des valeurs manquantes pour les données numériques
    num_imputer = SimpleImputer(strategy='median')
    num_train_imputed = num_imputer.fit_transform(num_train)
    num_test_imputed = num_imputer.transform(num_test)
    # Imputation des valeurs manquantes pour les données catégorielles
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_train_imputed = cat_imputer.fit_transform(cat_train)
    cat_test_imputed = cat_imputer.transform(cat_test)

    # Convertir les tableaux numpy en DataFrames après imputation
    num_train_imputed = pd.DataFrame(num_train_imputed, columns=num_train.columns)
    cat_train_imputed = pd.DataFrame(cat_train_imputed, columns=cat_train.columns)
    num_test_imputed = pd.DataFrame(num_test_imputed, columns=num_test.columns)
    cat_test_imputed = pd.DataFrame(cat_test_imputed, columns=cat_test.columns)

    # standardisation des variable numériques
    from sklearn.preprocessing import StandardScaler
    X_train.info()
    cols = ['WRank','LRank','Wsets','Lsets']
    sc = StandardScaler()
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])

    #Encodage  des variables catégorielle
    from sklearn.preprocessing import OneHotEncoder

    # Encodage des variables catégorielles
    encoder = OneHotEncoder(handle_unknown='ignore')
    cat_train_encoded = encoder.fit_transform(cat_train_imputed)
    cat_test_encoded = encoder.transform(cat_test_imputed)

    # Reconstitution des DataFrames après imputation et encodage
    cat_train_encoded = pd.DataFrame(cat_train_encoded.toarray(), columns=encoder.get_feature_names_out(cat_train.columns))
    cat_test_encoded = pd.DataFrame(cat_test_encoded.toarray(), columns=encoder.get_feature_names_out(cat_test.columns))
    # Concaténer les DataFrames pour obtenir les jeux d'entraînement et de test complets
    X_train_encoded = pd.concat([num_train_imputed, cat_train_encoded], axis=1)
    X_test_encoded = pd.concat([num_test_imputed, cat_test_encoded], axis=1)

    # affichage du Data Frame encodé
    X_train_encoded
    X_train_encoded.shape
    

if page == pages [4]:
  
  st.header("Machine learning")
  st.write("Cette section, présente les résultats obtenus lors de nos mélisations.")

  #reprise des éléments section précédente pour modélisation 
  
  #variable cible
  atp['target'] = (atp['elo_winner'] > atp['elo_loser']).astype(int)
  
  #supression des colonnes inutiles
  liste=['Comment','Date','Best of','Location','Tournament','PSW','PSL','B365W','B365L','Loser','Winner','ATP','elo_loser','elo_winner','proba_elo']
  atp=atp.drop(columns=liste,axis=1)
  
  #Séparation jeu test et entrainement
  from sklearn.model_selection import train_test_split
  X = atp.drop('target',axis=1)
  y = atp['target']

  # Diviser en jeux d'entraînement et de test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Séparer les colonnes numériques et catégorielles pour le jeu d'entraînement
  num_train = X_train.select_dtypes(include=['number'])
  cat_train = X_train.select_dtypes(include=['object'])
  # Séparer les colonnes numériques et catégorielles pour le jeu de test
  num_test = X_test.select_dtypes(include=['number'])
  cat_test = X_test.select_dtypes(include=['object'])

  #remplacement valeurs manquante avec le simple Imputer
  from sklearn.impute import SimpleImputer

  # Imputation des valeurs manquantes pour les données numériques
  num_imputer = SimpleImputer(strategy='median')
  num_train_imputed = num_imputer.fit_transform(num_train)
  num_test_imputed = num_imputer.transform(num_test)
  # Imputation des valeurs manquantes pour les données catégorielles
  cat_imputer = SimpleImputer(strategy='most_frequent')
  cat_train_imputed = cat_imputer.fit_transform(cat_train)
  cat_test_imputed = cat_imputer.transform(cat_test)
  # Convertir les tableaux numpy en DataFrames après imputation
  num_train_imputed = pd.DataFrame(num_train_imputed, columns=num_train.columns)
  cat_train_imputed = pd.DataFrame(cat_train_imputed, columns=cat_train.columns)
  num_test_imputed = pd.DataFrame(num_test_imputed, columns=num_test.columns)
  cat_test_imputed = pd.DataFrame(cat_test_imputed, columns=cat_test.columns)

  # standardisation des variable numériques
  from sklearn.preprocessing import StandardScaler
  X_train.info()
  cols = ['WRank','LRank','Wsets','Lsets']
  sc = StandardScaler()
  X_train[cols] = sc.fit_transform(X_train[cols])
  X_test[cols] = sc.transform(X_test[cols])

  #Encodage  des variables catégorielle
  from sklearn.preprocessing import OneHotEncoder
  # Encodage des variables catégorielles
  encoder = OneHotEncoder(handle_unknown='ignore')
  cat_train_encoded = encoder.fit_transform(cat_train_imputed)
  cat_test_encoded = encoder.transform(cat_test_imputed)

  # Reconstitution des DataFrames après imputation et encodage
  cat_train_encoded = pd.DataFrame(cat_train_encoded.toarray(), columns=encoder.get_feature_names_out(cat_train.columns))
  cat_test_encoded = pd.DataFrame(cat_test_encoded.toarray(), columns=encoder.get_feature_names_out(cat_test.columns))
  # Concaténer les DataFrames pour obtenir les jeux d'entraînement et de test complets
  X_train_encoded = pd.concat([num_train_imputed, cat_train_encoded], axis=1)
  X_test_encoded = pd.concat([num_test_imputed, cat_test_encoded], axis=1)

  # Modélisation

  # Utilisation des modèles de classifications
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import classification_report
  
  def prediction(classifier, X_train_encoded, y_train):
    if classifier == 'Random Forest':
      clf = RandomForestClassifier()
    elif classifier == 'Decision Tree':
        clf = DecisionTreeClassifier()
    elif classifier == 'Logistic Regression': 
        clf = LogisticRegression()
    else:
      raise ValueError("Classificateur non valide.")
    clf.fit(X_train_encoded, y_train)

    # Sauvegarde du modèle
    import joblib
    joblib.dump(clf, "model.pkl")
    return clf

  def scores(clf, choice, X_test_encoded, y_test):
    if choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test_encoded))
    elif choice == 'classification_report':
         y_pred = clf.predict(X_test_encoded)
         return classification_report(y_test, y_pred)
    else:
      raise ValueError("Choix non valide.")

  # Interface de choix/
  choix = ['Logistic Regression', 'Decision Tree', 'Random Forest']
  option = st.selectbox('**Choix du modèle**', choix)
  st.write('Le modèle choisi est :', option)

  #
  try:
    clf = prediction(option, X_train_encoded, y_train)

    # Sauvegarde du modèle
    import joblib
    joblib.dump(clf, "model.pkl")

    display = st.radio('Que souhaitez-vous montrer ?', ('Confusion matrix', 'classification_report'))

    if display == 'Confusion matrix':
        st.write("### Matrice de confusion :")
        st.dataframe(scores(clf, display, X_test_encoded, y_test))
    elif display == 'classification_report':
        st.write("### Rapport de classification :")
        st.text(scores(clf, display, X_test_encoded, y_test))
  except Exception as e:
        st.error(f"Une erreur est survenue : {e}")

  st.write("**Analyse de nos premiers résultats**:")
  st.write("Nos trois modèles montrent des performances intéressantes:")
  st.write(
  "\n- **La Régression Logistique**, de grosses faiblesses pour prédire la 0 mais excellente pour prédire la classe 1."
  "\n- **L'Arbre de Décision** se révèle être solide et équilibré avec des performances homogènes sur toutes les métriques."
  "\n- **Le Random Forest** se distingue comme notre modèle le plus performant, offrant une performance équilibrée et élevée sur toutes les classes.")
  


  st.write("**Variables importantes pour nos modèles:**")
  #les données les plus importantes de nos modeles

  with st.expander("Afficher les variables les plus importantes pour nos modèles"):
    #Décision tree
    dt_clf=DecisionTreeClassifier (random_state=42)
    dt_clf.fit(X_train_encoded,y_train)

    feat_importances = pd.DataFrame(data=dt_clf.feature_importances_, index=X_test_encoded.columns, columns=["Importance"])
    # Triage par importance
    feat_importances.sort_values(by="Importance", ascending=False, inplace=True)
    
    #création de la figure pour visualisation
    fig, ax = plt.subplots( figsize=(8,6))
    feat_importances.plot(kind='bar',ax=ax)
    ax.set_title("Importance des variables")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Importance")
    st.pyplot(fig)




  st.subheader("1/ Optimisation de nos modèles:")
  st.write("Après analyse des résultats de nos trois modèles, nous avons décidé d'optimiser les deux plus performants à savoir l’**Arbre de décisions** et le **Random Forest**.")
  st.write("Cette optimisation a été réalisée grâce à des **hyperparamètres** que nous avons déterminées grâce au **Grid Search avec validation croisée**.")
  st.write("Cette approche nous a permis de déterminer les meilleurs hyperparamètres pour améliorer la performance et la précision de nos modèles.")

  #Présentaiton des hyper paramètres
  def afficher_hyperparametres(modele):
    if modele == '':
       st.write("")
    elif modele == 'Decision Tree':
        st.write(" *Hyperparamètres pour Decision Tree* :")
        st.write("- **max_depth** : 5")
        st.write("- **min_samples_leaf** : 1")
        st.write("- **min_samples_split** : 2")
    elif modele == 'Random Forest':
        st.write(" *Hyperparamètres pour Random Forest* :")
        st.write("- **max_depth** : 15")
        st.write("- **min_samples_leaf** : 2")
        st.write("- **min_samples_split** : 20")
        st.write("- **n_estimators** : 50")
        st.write("- **max_features** : sqrt")
    else:
        st.error("Erreur : modèle inconnu.")

  # Options disponibles
  choix_hp = ['','Decision Tree', 'Random Forest']   
  option_hp = st.selectbox('**Sélectionnez un modèle pour voir ses hyperparamètres**', choix_hp)
  # Afficher le modèle choisi
  st.write('**Le modèle choisi est** :', option_hp)
  # Afficher les hyperparamètres associés
  afficher_hyperparametres(option_hp)
  

  # Résultats de nos modèles optimisés.
  st.subheader("2/ Résultats de nos modèles optimisés :")
  st.write("En appliquant ces hyperparamètres à nos modèles, nous obtenons les résultats suivants: ")

  # Meilleurs hyperparamètres pour les modèles
  meilleurs_parametres_dt = {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
  meilleurs_parametres_rf = {
    'max_depth': 15, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 2, 
    'min_samples_split': 20, 
    'n_estimators': 50
}

  # Fonction pour optimiser et entraîner un modèle
  def optimisation(optimisation, X_train_encoded, y_train):
    if optimisation == 'Decision Tree':
        modele = DecisionTreeClassifier(**meilleurs_parametres_dt)
    elif optimisation == 'Random Forest':
        modele = RandomForestClassifier(**meilleurs_parametres_rf)
    else:
        raise ValueError("Classificateur non valide.")
    
    modele.fit(X_train_encoded, y_train)
    return modele

  # Fonction pour afficher les scores
  def scores(modele, choice, X_test_encoded, y_test):
      if choice == 'Confusion matrix':
          y_pred = modele.predict(X_test_encoded)
          return confusion_matrix(y_test, y_pred)
      elif choice == 'Classification report':
          y_pred = modele.predict(X_test_encoded)
          return classification_report(y_test, y_pred)
      else:
          raise ValueError("Choix non valide.")

  # Interface utilisateur avec Streamlit
    # Sélection du modèle
  choix = ['Decision Tree', 'Random Forest']
  option = st.selectbox('**Choix du modèle**', choix)
  st.write('Le modèle choisi est :', option)

  try:
    # Optimisation et entraînement
    clf_opt = optimisation(option, X_train_encoded, y_train)

    # Choix de l'affichage
    display = st.radio('Que souhaitez-vous montrer ?', ['Confusion matrix', 'Classification report'])

    if display == 'Confusion matrix':
      st.write("### Matrice de confusion :")
      st.dataframe(scores(clf_opt, display, X_test_encoded, y_test))
    elif display == 'Classification report':
      st.write("### Rapport de classification :")
      st.text(scores(clf_opt, display, X_test_encoded, y_test))
  except Exception as e:
      st.error(f"Une erreur est survenue : {e}")

  st.write("**Analyse de nos modèles optimisé:**")
  st.write("\n- L'optimisation des hyperparamètres améliore considérablement les performances des deux modèles, **augmentant la précision, le rappel et le F1-score**.")
  st.write("\n- La robustesse et les résultats globalement supérieurs du **Random Forest** optimisé en font le choix idéal pour des prédictions fiables.")

  st.subheader("3/ Choix du model:")
  st.write("Le modèle **Random Forest est celui que nous avons choisi** en raison de sa meilleure performance globale et de son **exactitude de 82%**, surpassant les deux autre modeles.")
  st.write("Grâce à ses **gains équilibrés en précision, rappel, et F1-score sur toutes les classes**, il offre une **robustesse** accrue et une **fiabilité supérieure** pour des prédictions cohérentes, ce qui en fait le modèle idéal pour notre cas d'uasage.")

  st.subheader("4/ Comparaison de nos prédictions vs données initiales :")
  #Récupération des éléments réalisés plus haut pour créer y_pred

  # Création instance du modèle avec les hyperparamètres
  modele_rf_optimise = RandomForestClassifier(**meilleurs_parametres_rf)
  # Entraînement du modèle (supposant que X_train_encoded et y_train existent)
  modele_rf_optimise.fit(X_train_encoded, y_train)
  # Prédiction sur les données de test
  y_pred_rf = modele_rf_optimise.predict(X_test_encoded)

  # récupération des prédictions Arbre de décision
  df_comparaison_rf = pd.DataFrame({'Données_initiales (y_test)': y_test, 'Prédictions (y_pred)': y_pred_rf})
  #st.write(df_comparaison_rf.head(10))

  # Ajout colonne "Différence"
  df_comparaison_rf['Difference'] = df_comparaison_rf['Données_initiales (y_test)'] != df_comparaison_rf['Prédictions (y_pred)']
  # Comptage des erreurs totales
  nombre_differences = df_comparaison_rf['Difference'].sum()
  
  st.write ("**a/Echantillon:**")
  st.write((f"Nombre total de differences : {nombre_differences}"))
  st.write(df_comparaison_rf.reset_index(drop=False). head (12))
  # Voir les erreurs spécifiques
  #st.write((df_comparaison_rf[df_comparaison_rf['Difference']].head(10)))

  st.write("**b/Comparaison des valeurs de notre échantillon de tests et nos prédictions :**")
  
  # comptage du nombre d'occurences pour chaque valeurs des données initiales
  donnees_initiales_counts = df_comparaison_rf['Données_initiales (y_test)'].value_counts()
  # comptage du nombre d'occurences pour chaque valeurs des données prédites
  predictions_counts = df_comparaison_rf['Prédictions (y_pred)'].value_counts()

  # Creation subplots
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  # Plot the pie pour données initiales
  axes[0].pie(donnees_initiales_counts, labels=donnees_initiales_counts.index, autopct='%1.1f%%', startangle=90)
  axes[0].set_title('Données_initiales (y_test)')

  # Plot  pie pour prédiction
  axes[1].pie(predictions_counts, labels=predictions_counts.index, autopct='%1.1f%%', startangle=90)
  axes[1].set_title('Prédictions (y_pred)')
  # Affichage dans Streamlit
  st.pyplot(fig)

  st.write("Nos principaux écarts, observés à travers la matrice de confusion, proviennent de la **complexité pour notre modèle à prédire la victoire des outsiders (classe=0)**.")

  st.write("**c/Zoom sur les différences de nos prédictions:**")
  
  #création df avec les differences
  df_true = df_comparaison_rf[df_comparaison_rf['Difference'] == True]
  #distribution des prédictions
  distribution_predictions= df_true['Prédictions (y_pred)'].value_counts(normalize=True) * 100
  st.write("**Distribution des prédictions différentes des données initiales*")
  st.write(distribution_predictions)
  
  fig, ax = plt.subplots()
  labels = distribution_predictions.index
  sizes = distribution_predictions.values

  plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.55})
  plt.title("Répartition des prédictions différentes des données initiales")
  st.pyplot(fig)
  
  st.write("Cette visualisation renforce l'observation selon laquelle **la majorité des écarts provient des prédictions annonçant la victoire des favoris**, alors que ce sont des outsiders qui ont remporté la victoire en réalité.")



if page == pages [5]:
   image = r"C:\Users\nicol\Desktop\FORMATION DATA ANALYST\PROJET PARIS SPORTIF\tennis2.JPG"
   st.image(image, width=300)
   st.header ("Conclusion")

   st.write("Principaux enseignements de notre projet :")
   st.write("\n- **Le Random Forest** est le modèle le plus adapté à notre problématique, combinant **performance et résistance au surapprentissage**.")
   st.write("\n- Les variables clés influençant la victoire sont **le rang des joueurs et la surface du match**, des insights précieux pour les analystes sportifs.")
   st.write("\n- La comparaison avec les bookmakers ouvre des perspectives intéressantes : **notre modèle pourrait servir à détecter des cotes 'sous-évaluées'**, bien que des tests supplémentaires soient nécessaires pour valider cette approche.")
   st.subheader("Perspectives") 
   st.write("Pour optimiser nos résultats, plusieurs approches nous paraissent accessibles à l'exploration : "
   "\n- **Enrichissement** de  notre jeux de données : intégration d'autres données comme le ratio de points gagnés sous pressions (**Tie Break**), forme récente des joueurs (**serie de victoires en cours, retour de blessures**) ou encore **une variable temporelle** pour mesurer l'effet du temps sur les performances des joueurs."
   "\n- **Réechantillonnage**: Oversampling des matchs sérrés pour réduire les biais envers les favoris."
   "\n- Utilisation de **modeles plus complexes**: Réseaux de neurones, XGBoost."
   "\n- **Développement d'une interface** pour comparer visuellement nos prédictions avec les côtes du marché.")
