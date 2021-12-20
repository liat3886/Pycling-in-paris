

import pandas as pd
import streamlit as st
from PIL import Image 
import shap
import xgboost as xgb
from xgboost import plot_importance

st.set_page_config(layout="wide")
# st.image(img,caption='image credit "le monde"',)

pages = ['Résultats du projet',
         'Dataset',
         'Analyses',
         'Cartographie',
         'Prédiction du trafic',
         'Outil de prédictions']


page = st.sidebar.radio("Navigation", options=pages)


if page == "Résultats du projet":
    
    colx, coly, colz = st.columns((1,3,1))  
    
    with colx:
        st.write('')
    with coly:        
        img = Image.open("Photo_vélo.jpg") 
        st.image(img,caption = "Cyclistes à Paris pendant la période du Covid19, crédit https://www.francebleu.fr",use_column_width='always')
        st.title('Analyse du trafic cycliste à Paris',)
                
    with colz:
        st.write('')

    col10, col11, col12 = st.columns((1,3,1))  
    
    with col10:
        st.write('')
    with col11:
        st.info(
        """
        #### Projet Final de la formation DATA ANALYST chez [DataScientest](https://datascientest.com)   
        
        **Bootcamp Octobre 2021**
        
        **Auteurs :**
        
        * ###### Daniel Moutou | [Linkedin](https://www.linkedin.com/in/danielmoutou/)
    
        * ###### Tévy Phy | [Linkedin](https://www.linkedin.com/in/tévy-phy/)
        
        * ###### Liat Levi | [Linkedin](https://www.linkedin.com/in/liat-l)
        
        **Source de Données :**
        
        [Comptage vélo](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name) | [Données météo](https://www.historique-meteo.net/france/ile-de-france/paris/)
        
        [GitHub](https://github.com/DataScientest-Studio/Pycling)
        
        """)
      
        
    with col12:
        st.write('')
    
    


    

    
    st.write('')  
    
    st.header('I. Contexte')
    st.write('Depuis deux ans, les déplacements en vélo à Paris se sont accrus et ont explosé durant la crise sanitaire. Ils représentent aujourd’hui **5.6% des déplacements** vs 9% en voiture.')  
    st.write('Ainsi, en octobre dernier, la ville de Paris a annoncé un plan d’investissement de 250 millions d’euros pour pérenniser les « coronapistes », pistes cyclables qui ont vu le jour durant le confinement sur les grands axes parisiens ; ce qui démontre une volonté de massifier cet usage.')
    st.write('Mais le boom de la pratique du Vélib à Paris se poursuit-elle en 2021 ?')
    st.write('Nous étudierons l’évolution du trafic cycliste au travers de données mise à disposition par la Mairie de Paris sur la période du **1er Septembre 2020 au 31 Octobre 2021**.')
    
    
    st.header('II. Résultats')
    st.subheader('1. Les caractéristiques du trafic')
    st.write('L’usage du vélo à Paris se caractérise par :') 
    st.write('-	Septembre, le mois du Vélib à Paris, avec le trafic le plus important de l’année*. Septembre 2021 reste le mois à plus fort trafic, bien qu’une baisse d’utilisateurs de **-5.6%** soit enregistrée. C’est la baisse du trafic en semaine **-9%** qui explique ce recul, on peut supposer que le télétravail en soit responsable.')
    
    img = Image.open("Trafic_journalier2.png") 
    st.image(img, width = 800) 
    st.write('-	Grâce à l’ajout de notre variable d’Octobre 2021, cela nous a permis de constater qu’octobre 2021 a connu une augmentation de **+11.5%** de cyclistes, liée à une hausse d’usage en semaine **+14.3%**, sans doute dû au retour progressif sur site.')
   
    st.write("**VACANCES :**")
    st.write("-	Un trafic très sensible aux **vacances scolaires -27,4%**, ponts et **jours fériés -30,4%**")
    img = Image.open("Trafic Vacances scolaires.png") 
    st.image(img, width = 700)
   
    st.write("**WEEK-END :**")
    st.write('-	Une hausse du trafic le week-end est à noter sur septembre et octobre 2021, respectivement de **+9%** et **+8.4%**, ce qui laisse présager une progression continue de la pratique cycliste malgré les événements exceptionnels qu’ont connu la fin d’année 2020 et l’année 2021.')
    img = Image.open("Comparatif SO 2020-2021.png") 
    st.image(img, width = 700) 
    
    
    st.write("-	 Une forte disparité de trafic entre la semaine _(Weekend=0)_ : certains sites pouvant comptabilisés jusqu’à **18 000 cyclistes par heure** ; et le **week-end** _(Weekend=1)_ : **-29.4%**. On estime que c’est un **moyen de transport utilisé pour se rendre au travail**.")
    img = Image.open("Trafic semaine vs weekend.png")
    st.image(img, width = 800)
    
    st.write("**METEO :**")
    st.write("-	Le vélib est clairement un moyen de transport saisonnier, la courbe de trafic suit globalement celle des températures. Sans surprise, la pratique du vélib est conditionnée par la météo, notamment la pluie. A chaque journée pluvieuse, le trafic connaît une légère variation.")
    img = Image.open("Trafic Pluie.png")
    st.image(img, width = 700)   
    
    st.write("**PANDEMIE :**")
    st.write("-	Des événements exceptionnels sont venus ponctuer l’usage du vélo, comme **le couvre-feu -30% de trafic** ")    
    img = Image.open("Trafic couvre-feu.png")
    st.image(img, width = 700)   
        
    st.write("-	La crise sanitaire au travers du confinement a eu pour effet, un recul de **-19%** du trafic.")
    
    st.write("-	Lorsque nous observons la distribution du comptage par site et par heure, le trafic est caractérisé par des heures de pointes entre 6h et 7h et entre 16h et 17h, pouvant enregistrer plus de 25 000 cyclistes par heure pour certains sites. Nous nous étonnons de ces horaires que nous qualifions de « tôt » : démarrer sa journée plus tôt afin de rentrer avant le couvre-feu. ")
    img = Image.open("Distribution du trafic par heure.png")
    st.image(img, width = 700)  
    
    st.write("-	Cette année 2021 a été marquée par une période de **couvre-feux successifs du 15 décembre au 20 Juin**, nous souhaitons savoir si ceux-ci ont modifié le comportement du trafic cycliste. Observons la distribution du trafic par heure hors couvre-feu (Couvre_feu = 0) : cette fois-ci, les heures de pointes correspondent davantage à un trafic « habituel », à savoir un démarrage du trafic à partir de 7h et une baisse du flux à partir de 19h.")
    img = Image.open("Trafic hors couvre-feu.png")
    st.image(img, width = 700)     
     
    
    

    st.subheader('2. La prédiction du trafic')
    st.markdown("##### 1ère Itération")

    st.write("**Caractéristiques de cette 1ère itération :**")
    st.write("- Utilisation de 4 modèles classiques de régression linéaire")    
    st.write("- Normalisation des données")
    st.write("- Le modèle Lasso a retenu toutes les variables dans la prédiction")
    img = Image.open("1ère itération.png")
    st.image(img, width = 400)
    
    st.write("**Conclusion :** les performances des modèles ne sont pas satisfaisantes au travers de plusieurs axes.")
    st.write("")
    st.write("")
    
    st.markdown("##### 2ème Itération")    
    st.write("**Caractéristiques de cette 2ème itération :**")

    st.write("- Normalisation des données")
    st.write("- Test des ensembles d'entraînement et de test automatique avec la fonction 'train test split' et _shuffle=False_ pour garder la temporalité de notre jeu de données")
    st.write("- Ajout des variables _'comptage_horaire'_ H-1, H-2 et H-3")
    st.write("- Utilisation de la méthode SelectKbest pour la sélection des variables, avec k sélectionné arbitrairement")
    
    st.write("**Résultats de la 2ème itération :**")
    img = Image.open("2ème itération.png")
    st.image(img, width = 450)      

    st.write("**Conclusion :** Cette itération montre à quel point les variables sont importantes pour rendre un modèle de prédictions performant.")
    st.write("L'intégration du SelectKBest, nous a permis de connaître les 5 variables sur lesquelles s'est appuyé notre modèle pour effectuer la prédiction : _'Temperature_maxi'_, _'Arrondissement'_, _'Comptage h1'_, _'Comptage h2'_ et _'Comptage h3'_.")
    st.write("Nos performances sont nettement meilleures, le Gradient Boosting Regressor délivre de meilleures prédictions que les modèles simples. Ces modèles de prédictions linéaires complexes sont donc les plus adaptés à nos données. Dans la prochaine itération, nous verrons comment affiner nos performances.")
    st.write("")
    st.write("")
    
    st.markdown("##### 3ème Itération")
    st.write("**Caractéristiques de cette 3ème itération :**")
    st.write("- Utilisation d'object Pipeline")
    st.write("- Test des ensembles d'entraînement manuel")
    st.write("- Ajout des variables *'comptage_horaire'* J-1, J-2 et J-3 et S-1, S-2 et S-3")
    st.write("- SelectionKBest arbitraire")
    st.write("- Tunning des hyperparamètres")

    st.write("**Résultats de la 3ème itération :**")
    img = Image.open("recap.png")
    st.image(img, width = 450)
    
    st.write("**Conclusion :** le modèle XGBoost délivre les meilleures performances. Suite aux différentes tentatives de modifications de notre code, aux fuites involontaires de données engendrées par les répétitions de codes similaires pour chaque algorithme de Machine Learning, nous avons cherché à uniformiser notre code.")
    st.write("Pour cela, nous avons trouvé dans la Pipeline, la solution pour y remédier. Elle a également confirmé notre choix, contrairement aux précédentes itérations, de ne pas normaliser les données.")
    st.write("")
    st.write("")
 
    st.header('III. Conclusion et perspectives')
    st.write("Notre analyse du trafic nous a permis de comprendre les raisons des fluctuations du trafic sur 2020 et 2021. L'historique des mois de Septembre et Octobre 2021 vs 2020 présentent une tendance positive quant à l'évolution du trafic à venir*.")
    st.write("Néanmoins, deux mois nous semblent trop courts pour établir une conclusion générale. Nous préconisons d'importer les données disponibles des mois à venir afin de voir si cette progression se confirme sur le site de la [Mairie de Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name)., d'autant plus que le contexte sanitaire apporte des incertitudes et ralentit la tendance haussière du trafic.")
    st.write("Nous pensons qu'il y a une corrélation entre les données météorologiques et le trafic cycliste. Si nous avions eu plus de temps, nous les aurions intégrées au dataset afin de mesurer le réél impact de la météo heure par heure.") 
    st.write("Par ailleurs, nous avons pu lire dans de nombreux journaux la hausse des accidents de vélo liée à l'explosion de leurs usages. Les données de l'année 2020 venant d'être publiées le 24 Novembre dernier, nous pourrions étudier la corrélation entre ces accidents et l'utilisation du vélo à Paris. Ces accidents sont majoritairement causés par les véhicules à moteur, il serait donc intéressant d'enrichir le modèle avec les données du [trafic automobile](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/information/?disjunctive.libelle&disjunctive.etat_trafic&disjunctive.libelle_nd_amont&disjunctive.libelle_nd_aval) pour identifier les lieux accidentogènes et savoir si les voies cyclistes sont suffisament larges et adaptées à la hausse du trafic.")
    st.write("La granularité pourrait être également modifié pour analyser les données sur des périodes de la journée comme le matin, midi, soir et nuit, ou encore sur non pas une mais 2,3 voire 4 heures et étudier la performance de prédiction.")
if page == "Dataset":
    
    
    st.title('Le Dataset')
    st.subheader('1. Source')
    st.write("Notre jeu de données est issu du site de la Mairie de Paris disponible en Opendata pour évaluer le développement de la pratique cycliste : [Comptage Vélo] (https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name).")
    
    st.subheader('2. Fenêtre temporelle')
    st.write('Ce jeu de données présente l’ensemble des comptages vélo horaires sur 13 mois glissants (J-13 mois), mis à jour à J-1. Afin d’apporter une analyse comparative, nous avons récupéré les données complètes d’octobre 2021. Nous disposons de **14 mois de données** allant du **1er Septembre 2020 au 31 Octobre 2021 à partir** de 97 compteurs.')
    
    
    st.write('Notre variable cible sera le “Comptage horaire”, la seule variable numérique qui correspond au nombre de vélos/heure.') 
    st.write('Nous n’avons que deux variables explicatives à :') 
    st.write('- Dimension Temporelle : Date et heure de comptage,') 
    st.write('- Dimension Géographique : Coordonnées géographiques.')
    
    st.subheader('4. Données')
    img = Image.open("df.info.png") 
    st.image(img, width = 500)
       
           
    st.subheader('3. Préparation des données')
    st.write('**Langage utilisé :** Python')
    st.write('**Librairies utilisées :** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn')
    st.write('**Module :** Folium, Shap, XGBoost')
    st.write('**Taille du DataFrame :** 966 708 lignes x 9 colonnes')
    st.markdown('**Doublons :** aucun')
    img = Image.open("df_orig.png") 
    st.image(img, width = 500)
      
    st.write(" ")
    st.write(" ")
    st.write("**Valeurs manquantes :** L'_Identifiant technique compteur_ est la seule variable comportant des valeurs manquantes, qui représentent 0.7% du dataset. Nous avons décidé de la supprimer car nous n'utiliserons pas cette donnée.")
    st.write('**Suppression de variables (inutiles ou redondantes) :**')
    st.write("-	‘_Coordonnées géographiques_’ a été retraitée, séparée en 'Latitude' et 'Longitude'. Nous pourrons ainsi identifier les axes avec les plus gros trafics.")
    st.write("-	‘_Identifiant du compteur_’ ")
    st.write("-	‘_Identifiant du site de comptage_’ ")
    st.write("-	‘_Nom du site de comptage_’ ")
    st.write("**Ajout de variables :**")
    st.write("Afin d’enrichir notre dataset, nous avons créé 21 nouvelles variables à partir de :")
    st.write("Source interne")
    st.write("-	Période : Jour, jour de la semaine, Mois, Année, Mois-année, Week-end")
    st.write("-	Congés : Vacances scolaires, Jours fériés")
    st.write("-	Géographie : Arrondissement")
    st.write("Source externe")
    st.write("-	Météorologique : Température (Minimum, maximum, moyenne), Pluie en mm à partir du site [meteoblue.com] (https://www.meteoblue.com/fr/meteo/historyclimate/weatherarchive/paris_france_2988507?fcstlength=1m&year=2021&month=1)")
    st.write("-	Evènements exceptionnels : Confinement, couvre-feu.")
  


if page == "Analyses":
    st.title("L'analyse du trafic cycliste à Paris par la Data Visualization")
    options = ['Evolution globale du trafic',
               'Analyse périodique',
               'Evénements cycliques',
               'Evénements exceptionnels']
      
    st.write("#### Analyse temporelle")
    option = st.radio('', options = options)
    
    if option == "Evolution globale du trafic":
        st.write("##### Evolution globale")
        img = Image.open("Trafic_journalier2.png") 
        st.image(img, width = 800)                 
        st.write("La fin d'année 2020 et l'année 2021 ont été ponctués par divers événements, qui ont eu pour effet un trafic très fluctuant.")
              
    if option == "Analyse périodique":
        st.write("#### Analyse périodique")  
        périodicités = ['Mensuelle', 'Hebdomadaire', 'Horaire']
        périodicité = st.radio('', options = périodicités)
        
        if périodicité == "Mensuelle":
            st.write("#### Analyse comparative Septembre/Octobre 2021 vs 2020")
            img = Image.open("Evolution mensuelle.png") 
            st.image(img, width = 800)           
            st.write("Le trafic a augmenté de +5.6% entre Septembre/Octobre 2021 vs 2021. ")                       
            st.write("Nous allons analyser ce qui contribue à cette hausse de trafic.")
            img = Image.open("Comparatif SO 2020-2021.png") 
            st.image(img, width = 700)
            st.write('-	Une hausse du trafic le week-end est à noter sur septembre et octobre 2021, respectivement de +9% et +8.4%, ce qui laisse présager une progression continue de la pratique cycliste malgré les événements exceptionnels qu’ont connu la fin d’année 2020 et l’année 2021.')       
    
        if périodicité == "Hebdomadaire":
            img = Image.open("Trafic semaine vs weekend.png") 
            st.image(img, width = 700)          
            st.write("Le trafic connaît une variation de -30% entre l'usage du vélo en semaine vs le week-end. On peut donc supposer que ce moyen de déplacements est utilisé pour se rendre au travail.")
            
        if périodicité == "Horaire":
            img = Image.open("Distribution du trafic par heure.png")
            st.image(img, width = 700)
            st.write('-	Une hausse du trafic le week-end est à noter sur septembre et octobre 2021, respectivement de +9% et +8.4%, ce qui laisse présager une progression continue de la pratique cycliste malgré les événements exceptionnels qu’ont connu la fin d’année 2020 et l’année 2021.')

    
    if option == "Evénements cycliques":
        st.write("#### Evénements cycliques")  
        cycliques = ['Vacances scolaires', 'Jours fériés', 'Météo']
        cyclique = st.radio('', options = cycliques)        
        
        if cyclique == "Vacances scolaires":
            img = Image.open("Trafic Vacances scolaires.png")
            st.image(img, width = 700)
            st.write("-	Un trafic très sensible aux vacances scolaires **-27,4%**.")
            
        if cyclique == "Jours fériés":
            img = Image.open("Jours Fériés.png")
            st.image(img, width = 250)
            st.write("-	Les jours fériés se traduisent par baisse de trafic de **-30,4%**.")

        if cyclique == "Météo":
            img = Image.open("Trafic Pluie.png")
            st.image(img, width = 700) 
            st.write("-	Le vélib est clairement un moyen de transport saisonnier, la courbe de trafic suit globalement celle des températures. Sans surprise, la pratique du vélib est conditionnée par la météo, notamment la pluie. A chaque journée pluvieuse, le trafic connaît une légère variation.")

    if option == "Evénements exceptionnels":
        st.write("#### Evénements exceptionnels")  
        exceptionnels = ['Confinements', 'Couvre-feu']
        exceptionnel = st.radio('', options = exceptionnels)        
        
        if exceptionnel == "Confinements":
            img = Image.open("Trafic Confinement.png")
            st.image(img, width = 300)         
            st.write("-	La crise sanitaire au travers du confinement a eu pour effet, un recul de -19% du trafic.")
        
        if exceptionnel == "Couvre-feu":
            st.write("-	Des événements exceptionnels sont venus ponctuer l’usage du vélo, comme le couvre-feu **-30% de trafic**")
            img = Image.open("Trafic couvre-feu.png")
            st.image(img, width = 700)  


if page == "Cartographie":
    st.title('Cartographie')
    st.subheader('Cartographie : l’intensité du trafic par localité')

    st.write("En utilisant le module Folium, nous avons placé sur la carte de Paris les emplacements géographiques des compteurs de passage horaire de vélo présents dans notre dataset.")
    img = Image.open("Emplacement compteurs.png")
    st.image(img, width = 700)     
    
    st.write("Sur ce 2ème graphique nous observons les comptages moyens sur chaque compteur, avec une couleur et une taille des halos liés à la valeur de ces comptages.")
    st.write("Nous observons logiquement les plus grands passages sur les grands axes parisiens, dans le plein centre aux alentours du quartier du Châtelet, des bords de Seine et des grandes gares.")
    st.write("Le compteur le plus fréquenté se situe au 73 Boulevard de Sébastopol, en plein centre, tandis que le moins fréquenté est au 28 Boulevard Diderot avec en moyenne 2 passages par heure.")
    img = Image.open("Intensité trafic par localité.png")
    st.image(img, width = 700) 
    
    st.write("Il est aisé de voir les grandes disparités de passages entre les 5 compteurs les plus fréquentés et les 5 compteurs qui le sont le moins sur les 2 graphiques ci-dessous, avec des moyennes journalières en heures de pointe entre 300 et 650 passages contre 2 à 17 passages :")
    
    img = Image.open("Top 5 compteurs.png")
    st.image(img, width = 700)    
    
    img = Image.open("Flop 5 compteurs.png")
    st.image(img, width = 700)        
    
    st.write("Suite à ces constats et aux nombreuses perturbations du trafic durant la période d’étude, nous décidons d’étudier leur corrélation sur le comptage horaire journalier et d’essayer de prédire ces derniers.") 
    st.write("Nous utiliserons pour cela le Machine Learning et sélectionnerons le modèle le plus précis quant aux comptages réels obser-vés et les comptages prédits par les algorithmes.")


if page == "Prédiction du trafic":
    st.title('La prédiction du trafic')
    st.header('Machine Learning : la prédiction du trafic grâce aux modèles prédictifs')
    
    st.subheader('Introduction')
    st.write('Après avoir identifié certaines corrélations entre nos variables et le comptage horaire par heure, jour et site, nous souhaitons utiliser le Machine Learning pour entraîner un modèle de prédiction de ces comptages')
    st.write('Notre variable cible est une variable quantitative. Les modèles qui correspondent à notre jeu de données sont donc de type régression linéaire. ')
    st.write('Nous utiliserons toutes les variables de notre dataset hormis "Date_heure_comptage" et "Date" qui sont de type datetime.time')
    st.write('En premier lieu, nous devrons faire un preprocessing')
 
    st.subheader('1. Preprocessing')
    st.write("Nous commençons par créer un nouveau DataFrame qui est essentiellement une copie de notre DataFrame d'origine, sur lequel nous allons travailler." )
    st.write("Afin d'appliquer un modèle Machine Learning, nos variables catégorielles doivent être encodées. Pour ce faire, nous utilisons le 'LabelEncoder'. ")
    st.write("Ensuite nous divisons notre jeu de donnée en 2 sets, un d'entrainement et un de test. Notre jeu de donnée étant temporel, il n'est pas pertinent d'utiliser un 'train_test_split' aléatoire qui ne prendrait pas en compte les tendances de chaque jour, mois,")
    st.write("Nous faisons donc le split manuellement, en entraînant le modèle sur les 24 premiers jours du mois et en gardant le reste pour le set de test. ")
    st.write("Nous définissons ensuite notre variable cible 'Comptage_horaire' et séparons nos features de cette dernière.")
    st.write ("Afin d'évaluer les performances du modèle, nous utiliserons la métrique RMSE. ")
    st.write("La Root Mean Square Error est l'écart type des résidus (erreurs de prédiction). Elle mesure la concentration des données autour de la 'line best fit'. En d'autres termes, cette mesure nous présente l'erreur de prédiction obtenue à partir du modèle, et décrit les différences entre les valeurs existantes et les valeurs attendues.")
   
    st.subheader('2.1ère Itération ')
    st.write('Nous souhaitons tester 4 modèles classiques de régression linéaire ')
    st.write('Régression linéaire')
    st.write('Lasso')
    st.write('Ridge')
    st.write('ELASTIC NET ')
    st.write("Etant donné la différence de variance entre nos variables, nous avons décidé de les normaliser à l'aide d'un StandardScaler." )
    st.write("Après avoir fait tourner les modèles, nous nous apercevons que les scores obtenus sont faibles. L'indice RMSE est bien trop élevé (environ 76-80) et nous obtenons des prédictions de comptage horaire négatifs, ce qui n'a aucun sens." )
    st.write('**score:**')
    img = Image.open("1ère itération.png") 
    st.image(img, width = 400)
    img = Image.open("variable_lasso.png") 
    st.image(img, width = 800)
    st.write('Toutes les variables sont bonnes à prendre selon Lasso. A priori car aucune n’est particulièrement corrélée à la target.')
    st.write('Conclusion :  ')
    st.write("Les modèles doivent donc être optimisés. Nous proposons de donner de la substance à notre modèle avec un effet de 'mémoire' en ajoutant les valeurs du comptage des passages à h-1, 2 et 3 afin de l'entraîner plus efficacement.")
   
    st.subheader('3.2ème Itération')
    st.write('Nous créons les 3 variables mentionnées en itération 1 : comptage horaire en h-1, h-2 et h-3.')         
    st.write("N'ayant pas de comptages horaires h-1, 2 et 3 au 1er jour de notre jeu de données, nous remplaçons les NaN par des 0. Le nombre de NaN est dérisoire comparé au nombre de lignes de notre dataset. Nous faisons donc l'hypothèse que ces valeurs à 0 n'impacteront pas les résultats. ")
    img = Image.open("Itération2_1.png")
    st.image(img, width = 800) 
    img = Image.open("Itération2_2.png") 
    st.image(img, width = 800)  
    st.write("Nous nous sommes également rendus compte que les comptages négatifs prédits en itération 1 venaient du fait que nous n'avions pas normalisé les sets de notre variable cible. Nous avons donc corrigé le problème en utilisant notre StandardScaler sur ces derniers.")
    st.write("A ce stade, nous avons commencé à examiner s'il était nécessaire de réduire le nombre de variables explicatives. ")
    st.write("Nous avons donc créé l'ACP mais les résultats n'étaient pas concluants pour l'analyse : nous avons 4 voire 5 composantes principales identifiées avec la méthode du coude. Nous avons représenté nos variables sur un cercle de corrélation à 2 composantes mais les résultats ne sont pas exploitables, ou du moins nous ne savions pas comment faire malgré nos recherche")
    st.write("Nous avons donc observé la corrélation entre les variables et notre cible dans l'ensemble de données avec une heatmap : nos nouvelles variables de comptage sont effectivement corrélées, contrairement à toutes les autres variables. ")
    img = Image.open("corr.png")
    st.image(img, width = 200)
    st.write("Pour cette raison, nous avons pensé qu'utiliser le SelectKBest nous permettrait de mettre en évidence les effets d'une sélection réduite de variables explicatives sur la performance du modèle : nous avons itéré nos modèles avec différentes valeurs de k, et avons constaté que nous obtenions les meilleurs résultats lorsque k = 10. Nos premières impressions suite à l'étude de la heatmap se sont vu confirmées")
    st.write("En supplément de cela, nous avons testé la division des sets de test et d'entrainement automatique : nous avons utilisé la fonction 'train_test_split' et un test_size = 0.2 avec l'argument shuffle = False pour conserver la temporalité de notre jeu de données." )
    st.write('Les résultats se sont avérés décevants, nous sommes donc revenus au split manuel.')
    st.write('Nous avons ensuite entraîné 3 nouveaux modèles :')
    st.write('DecisionTreeRegressor ')
    st.write('RandomForestRegressor  ')
    st.write('GradientBoostingRegressor ')
    st.write('Conclusion :  ')
    st.write("Les résultats n'étaient pas particulièrement concluants. Par ailleurs nous avons involontairement créé des problèmes de fuite de données dans notre jeu de données en entraînant, ajustant et transformant plusieurs fois nos données sur chaque modèle. Le code n'était pas optimisé et les résultats non plus." )
    st.write("Nous décidons donc d'utiliser des pipelines pour optimiser notre code en 3ème itération. Nous pensons également ajouter d'autres variables pour faire jouer l'effet de mémoire des modèles." )
   
   
    st.subheader('4.3ème Itération ')
   
    st.write('Nous créons les variables comptage horaire j-1, 2 et 3 ainsi que s-1, 2 et 3. Ayant cette fois un nombre de NaN conséquent, nous utilisons un fillna pour remplacer ces NaN par les valeurs moyennes respectives/compteur/jour/heure. ')
    st.write("Après ajout des nouvelles variables nous avons également réitéré des tests de SelectKBest et nous passons de k=10 à k=12 pour les meilleurs de nos modèles Ridge, Lasso et ElasticNet")
    st.write("Nous utilisons cette fois l'objet PIPELINE. Le pipeline Machine Learning est un moyen d'automatiser le flux de travail d'apprentissage en permettant aux données d'être transformées et corrélées en un modèle qui peut ensuite être analysé pour obtenir des résultats." )
    st.write('Ces pipelines rendent le processus de saisie des données dans le modèle Machine Learning clair et facilement reproductible sur les différents modèles. ')
    st.write("Nous intégrons dans ces pipelines nos modèles, avec le sélecteur de variables SelectKBest, le scaler StandardScaler, ainsi que les hyperparamètres à tester à l'aide d'une grille de recherche et d'une validation croisée. Cette méthode est bien plus efficace puisque les modèles itèrent sur les hyperparamètres et choisissent la meilleure combinaison. Attention cependant aux temps de chargement qui évoluent proportionnellement aux nombres de paramètres testés : nous choisissons de tester nos hyperparamètres 2 par 2 ou 3 par 3 maximum pour cette raison et nous implémentons la magic line %%time pour récupérer le wall-time de nos modèles")
    st.write("Nous utilisons les pipelines sur nos modèles regressor et sur la régression, qui nous retournent les meilleurs hyperparamètres, la meilleure moyenne de RMSE en validation croisée ainsi que le RMSE de notre jeu de test. ")
    st.write('Nous décidons d’essayer un AdaBoosting sur le DecisionTreeRegressor afin que les mauvaises prédictions soient systématiquement retravaillées à chaque niveau de branche. Cependant cela n’a pas changé notablement les résultats, nous en concluons donc que cet essai n’était pas pertinent. ')
    st.write("Nous décidons également de remplacer le Gradient Boosting Regressor par le XGBoost Regressor qui possède plus d’hyperparamètres et qui peut donc être affiné plus facilement. Attention au réglage de ces hyperparamètres qui augmente considérablement le temps de modélisation. Nous nous sommes contentés de tenter des combinaisons de 3 hyperparamètres, pour des rendus d’environ 2h. Nous aurions certainement pu améliorer les scores mais un test de rendu avec 4 hyperparamètres nous a donné une amélioration d’environ 8% pour près de 8h de rendu. Ratio temps/amélioration que nous n’avons pas jugé admissible.")
    st.write('Pour les autres modèles, nous considérons que la validation croisée qui sélectionne le meilleur alpha est suffisante.' )
    st.write("Les pipelines recommandent systématiquement de ne pas normaliser les données. Nous décidons donc d'abandonner notre StandardScaler pour nos modèles simples (nous faisons l'hypothèse que cela vient du fait que la différence max de variance de 80-90 entre nos variables identifiée avec df.describe() n'est pas contraignante sur un dataset de 700K+ lignes). ")
    st.write("Nous obtenons nos meilleurs résultats, avec des RMSE aux alentours de 21-27 (ce qui est très loin des 79-80 de la 1ère itération). ")
   
    img = Image.open("recap.png")
    st.image(img, width = 800)
   
    st.subheader('5.Etude du Meilleur Modèle   ')
    img = Image.open("xgb.png")
    st.image(img, width = 400)
    st.write("Après lancement de nos pipelines, le modèle XGBoost nous fournit les meilleurs résultats (RMSE = 21.834). Les autres modèles Regressor n'étant pas loin derrière et nos modèles simples non plus." )
    st.write("XGBoost est un modèle, basé sur des arbres de décision qui démontre la supériorité sur l'apprentissage automatique profond." )
    st.write("Cette direction et ce modèle nous ont fourni la plus petite erreur de prédiction pour le test et le groupe d'entrainement")
    st.write("De plus, l'indice R2 qui mesure la force de la relation entre le modèle et la variable dépendante qui lui est donnée : plus le R2 est élevé, plus le modèle est adapté pour prédire les observations. Dans notre modèle nous obtenons R2 = 0,933, ce qui signifie de bonnes prédictions.")
    img = Image.open("Comptage_top.png")
    st.image(img, width = 800)
    st.write("Afin de confirmer notre sélection de modèle, nous avons créé un graphique montrant le comptage prédit en moyenne sur le site avec la fréquence de passage la plus élevée (Totem 73 boulevard de Sébastopol S-N). Comme nous l'avions prévu, les prévisions moyennes sont très proches des données réelles")   
    img = Image.open("Comptage_flop.png")
    st.image(img, width = 800)
    st.write("Sur le site avec l'une des fréquences de passage la plus faible (28 boulevard Diderot E-O), nous avons constaté que la prédiction est bien moins précise")   
    st.write("Nous avons également voulu analyser la prédiction sur une journée au hasard sur le site le plus fréquenté, afin d’obtenir une vision plus précise qu’un comptage horaire moyen sur toute la période présente sur le dataset de test.")
    st.write("Nous avons donc choisi une date présente dans ce dernier, à savoir le 30/06/21, de manière complètement aléatoire")
    img = Image.open("Comptage_aléatoire.png")
    st.image(img, width = 800)
    st.write("Nous observons que notre modèle XGB est fiable en période creuse. En dehors de cela, nous retrouvons des écarts qui fluctuent selon les heures de la journée et qui semblent osciller entre notre fenêtre de RMSE = 22 et une 50aine de passages")
    st.write("Nous faisons l’hypothèse que les grandes variations de comptage par heure et par jour en période pleine donnent du fil à retordre au modèle. ")
    st.write("Nous en concluons que si le modèle est en moyenne performant, il y a des disparités selon le jour, l’heure mais également la fréquentation générale du compteur qui perturbent la qualité de prédiction. ")
    st.write("Pour comprendre cela, nous avons utilisé SHAP, qui permet d'identifier les variables qui participent le plus à l'apprentissage du modèle. ")
    st.write("Avec l'aide de SHAP nous essaierons d'expliquer les résultats obtenus et d'utiliser l'outil, afin de localiser les caractéristiques qui ont le plus contribué à la classification et comment elles ont contribué : ")
    img = Image.open("shap1.png")
    st.image(img, width = 800)
    st.write('Sur ce premier graphe : ')
    st.write("La position sur l'axe des abscisses indique une prédiction plus petite (à gauche) et plus grande (à droite) que la variable cible réelle. ")
    st.write("La couleur rouge signifie que la valeur prédite concerne une valeur cible réelle (un comptage horaire réel) élevée avec un impact positif sur la prédiction, la couleur bleue signifie que la valeur prédite concerne une valeur cible réelle (un comptage horaire réel) faible avec un impact négatif sur la prédiction. ")
    st.write("Si l'on observe les graphes de comptage prédit et réel on observe ce phénomène : les gros comptages horaires sur le site le plus fréquenté affectent positivement la prédiction, tandis que les petits comptages sur l'autre site affectent négativement cette dernière. ")
    img = Image.open("shap2.png")
    st.image(img, width = 800)
    st.write("Sur ce 2ème graphe : ")
    st.write("On comprend qu'il y a une relation linéaire positive entre la variable comptage_j1 et notre comptage horaire, et que la variable comptage j1 interagit le plus avec la variable comptage h2 que les hautes et faibles valeurs impactent positivement et faiblement") 

    st.subheader('Conclusion et perspectives ')
    st.write("Suite à l’analyse sur le trafic au travers de la data visualisation, l’année 2021 a été extrêmement perturbée par le confirnament et les couvre-feux successifs, le comparatif des mois de Septembre et octobre 2021 avec 2020 nous montrent une croissance sur le trafic semaine et week-end. Il faudrait que nous puissions récupérer d’autres données afin d’élargir cette analyse comparative.")
  
  
if page == "Outil de prédictions":  
    
    from datetime import date, time, datetime as dt
    
    from sklearn.feature_selection import SelectKBest, f_regression
    
    from sklearn.pipeline import Pipeline
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
    
    from sklearn.linear_model import RidgeCV, Ridge, LinearRegression, ElasticNetCV, ElasticNet, Lasso, LassoCV
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, GridSearchCV
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    
    df_2020 = pd.read_pickle("pickled_df20.pkl")
    df_2021 = pd.read_pickle("pickled_df21.pkl")
    df_ml_2020 = pd.read_pickle("pickled_df_ml20.pkl")
    df_ml_2021 = pd.read_pickle("pickled_df_ml21.pkl")
    
    df_ml_2020['Heure'] = df_ml_2020['Heure'].apply(lambda x: int(x.strftime('%H')))
    df_ml_2021['Heure'] = df_ml_2021['Heure'].apply(lambda x: int(x.strftime('%H')))
    df_2020['Heure'] = df_2020['Heure'].apply(lambda x: int(x.strftime('%H')))
    df_2021['Heure'] = df_2021['Heure'].apply(lambda x: int(x.strftime('%H')))
    
    
    st.header('MACHINE LEARNING')
    st.subheader('1. Paramétrage')
    st.markdown('**1.1. Choix des Données***')
    st.markdown('*' 'Choisir UN unique jeu de données')
    
    
    #--------Paramétrage des modèles
    
    with st.sidebar.subheader('Paramètres du TEST Set'):
        test_limit = st.sidebar.slider(label = "Choisir le jour du mois de démarrage du jeu de test", min_value = 15, max_value = 31, step = 1 )
        
    with st.sidebar.subheader('Hyperparamètres communs (Regressor)'):
        
        param_n_estimators = st.sidebar.slider('n_estimators', 0, 1000, 100, 50)
        param_max_features = st.sidebar.select_slider('max_features', options=['auto', 'sqrt', 'log2'])
        param_max_depth = st.sidebar.slider("Profondeur maximum de l'arbre (max_depth)", 1,9,7,1)
        param_random_state = st.sidebar.slider('random_state', 0, 500, 69, 1)
        param_criterion = st.sidebar.select_slider('Mesure de la performance (criterion)', options=['squared_error', 'absolute_error'])
        param_n_jobs = st.sidebar.select_slider('Nombre de jobs travaillant en parallèle (n_jobs)', options=[1, -1])
     
    with st.sidebar.subheader('Spécifiques XGBoost'):
        param_colsample_bytree = st.sidebar.select_slider("Sous-échantillonnage de l'arbre (col_sample_bytree)", options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        param_subsample = st.sidebar.select_slider("Fraction des observations randomisées pour chaque arbre(subsample)", options=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        param_eta = st.sidebar.select_slider("Réduction du poids des features (eta)", options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])    
        param_tree_method = st.sidebar.select_slider("Méthode de construction de l'arbre (tree_method)", options=['auto', 'approx', 'hist', 'gpu_hist'])
     
    with st.sidebar.subheader('Spécifiques DecisionTree'):
        param_splitter = st.sidebar.select_slider("Splitter", options=['best','random'])
        
    with st.sidebar.subheader('Spécifiques Régression Linéaire'):    
        param_normalize = st.sidebar.select_slider("Normalize", options=['True','False'])
        param_fit_intercept = st.sidebar.select_slider("Fit_intercept", options=['True','False'])
        
    
    #------------ SPLIT-------------------
    

    
   
    
    
    df_choisi=[]
    

    if st.checkbox('Données de Septembre-Octobre 2020', value=True):
       
        X_train = df_ml_2020[df_ml_2020['Jour'] <= test_limit]
        X_test = df_ml_2020[df_ml_2020['Jour'] > test_limit]
        y_train = df_ml_2020[df_ml_2020['Jour'] <= test_limit]
        y_test = df_ml_2020[df_ml_2020['Jour'] > test_limit]
        
        X_train = X_train.drop(['Comptage_horaire','Date_heure_comptage','Date'], axis = 1)    
        X_test = X_test.drop(['Comptage_horaire','Date_heure_comptage','Date'], axis = 1)                          
        y_train = y_train[['Comptage_horaire']]    
        y_test = y_test[['Comptage_horaire']]
        
        df_choisi = df_2020
        df_choisi = df_choisi[df_choisi['Jour'] > test_limit]
        
        st.success('Données Septembre-Octobre 2020 chargées...')
 


    if st.checkbox("Données de Juillet-Octobre 2021"):
        df_choisi = df_2021
        X_train = df_ml_2021[df_ml_2021['Jour'] <= test_limit]
        X_test = df_ml_2021[df_ml_2021['Jour'] > test_limit]
        y_train = df_ml_2021[df_ml_2021['Jour'] <= test_limit]
        y_test = df_ml_2021[df_ml_2021['Jour'] > test_limit]
        
        X_train = X_train.drop(['Comptage_horaire','Date_heure_comptage','Date'], axis = 1)    
        X_test = X_test.drop(['Comptage_horaire','Date_heure_comptage','Date'], axis = 1)                          
        y_train = y_train[['Comptage_horaire']]    
        y_test = y_test[['Comptage_horaire']]
        
        df_choisi = df_2021
        df_choisi = df_choisi[df_choisi['Jour'] > test_limit]
    
        st.success('Données Juillet-Octobre 2021 chargées...')  
    
        
    col1, col2 = st.columns((1,1))
        
    
    with col1:
        
        st.markdown('**1.2. Features**')
        st.info(list(X_train.columns))
        st.markdown('**1.3. Target**')
        st.info(list(y_train.columns))
        st.markdown('**1.4. Data split**')
        st.write('Training set')
        st.info(X_train.shape)
        st.write('Test set')
        st.info(X_test.shape)

       
    with col2:
         
        st.markdown('**1.4. Sélection des Features**')
        with st.echo(): 
                
            sk = SelectKBest(f_regression, k=12)
            sk.fit(X=X_train, y=y_train)
            sk_train = sk.transform(X_train)
            sk_test = sk.transform(X_test)
        
        st.write('Sélection des 12 meilleures features pour les modèles Ridge, Lasso, ElasticNET et Régression Linéaire ')
        st.write('k = 12 trouvé de manière empirique sur base des scores des modèles pour différentes valeurs de k ')
        st.write(X_train.columns[sk.get_support()])
        
    
        
    
    #--------Création de la fonction de choix du modèle ML
    
    st.subheader('2. Peformance du Modèle')
    st.markdown('**2.1. Choix du Modèle**')       
    modele_choisi = st.selectbox('Choisir le modèle de Machine Learning à tester', ['RidgeCV','Lasso','ElasticNET CV','Régression Linéaire', 'XGBoost Regressor','Random Forest Regressor', 'Decision Tree Regressor'])
    if modele_choisi == 'Random Forest Regressor':
        st.write("*Le Random Forest peut être un peu lent. Profitez-en pour vous étirer un peu si vous êtes resté assis longtemps :)")
    st.markdown('**2.2. Choix du compteur**')
    liste_compteurs = df_2021['Nom_compteur'].unique()
    compteur_choisi = st.selectbox('Choisir le site de comptage', liste_compteurs)
    st.markdown('**2.3. Choix du mois de prédiction**')
    
    liste_mois = df_choisi['Mois_annee'].unique()
    mois_choisi = st.selectbox('Choisir le mois de prédiction', liste_mois)
    
    
    
    def build_model(modele_choisi):
           
        if modele_choisi == 'XGBoost Regressor':
            model = xgb.XGBRegressor(n_estimators=param_n_estimators,
                colsample_bytree = param_colsample_bytree,
                subsample = param_subsample,
                eta = param_eta,
                random_state = param_random_state,
                max_features = param_max_features,
                max_depth = param_max_depth,
                criterion=param_criterion,
                n_jobs=param_n_jobs,
                tree_method = param_tree_method)
        
        if modele_choisi == 'Decision Tree Regressor':
            model = DecisionTreeRegressor(splitter = param_splitter,
                max_features = param_max_features,
                max_depth = param_max_depth)
            
        if modele_choisi == 'Random Forest Regressor':
            model = RandomForestRegressor(n_estimators=param_n_estimators,
                max_features = param_max_features,
                max_depth = param_max_depth,
         #       criterion = param_criterion,
                n_jobs=param_n_jobs)
             
                 
        if modele_choisi == 'Régression Linéaire':
            model = LinearRegression(normalize = param_normalize,
                n_jobs=param_n_jobs,
                fit_intercept = param_fit_intercept)
            
        
        if modele_choisi == 'RidgeCV':
            model = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
    
            
        if modele_choisi == 'Lasso':
            model = LassoCV(alphas = (10, 1, 0.1, 0.001, 0.0005), cv=5)
    
            
        if modele_choisi == 'ElasticNET CV':
            model = ElasticNetCV(l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
                            alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0), 
                            cv = 8, tol = 0.1)
            
            
        model.fit(sk_train, y_train)
        y_pred_train = model.predict(sk_train)
        y_pred_test = model.predict(sk_test)
        
        df_test = df_choisi
        df_test = df_choisi.sort_values(['Nom_compteur','Date','Heure'])
        df_test['Comptage_test'] = y_test
        df_test['Comptage_predict'] = y_pred_test.astype(int)
        df_test = df_test.loc[(df_test['Nom_compteur'] == compteur_choisi) & (df_test['Mois_annee'] == mois_choisi)].groupby(['Heure']).agg({'Comptage_test':'mean','Comptage_predict':'mean'})
        
        
        
        col3, col4 = st.columns((1,3))
        
        with col3:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write('SCORE TRAIN',model.score(sk_train, y_train))
            st.write('SCORE TEST',model.score(sk_test, y_test))
            st.write('RMSE TRAIN',mean_squared_error(y_train, y_pred_train, squared = False))
            st.write('RMSE TEST',mean_squared_error(y_test, y_pred_test, squared = False))
              
        with col4:
            st.write("Prédictions horaire du site '",compteur_choisi,"' pour la période ",mois_choisi)
            st.line_chart(df_test)
            
            
        return  ":)"
    
    #--------Affichages des résultats du modèle ML
    
    st.subheader('3. Résultats')    
    st.write(build_model(modele_choisi))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# https://medium.com/swlh/a-beginners-guide-to-streamlit-5e0a4e711968
#st.sidebar.header("Features")
#st.sidebar.markdown("Drag the sliders")

#row = st.sidebar.slider("Display Records:", 0, 100, 50)

# https://blog.octo.com/creer-une-web-app-interactive-en-10min-avec-streamlit/




 #" streamlit-folium"

#with st.echo():
   # import streamlit as st
    #from streamlit_folium import folium_static
    #import folium

#Création d'un dataframe pour la création de la carte représentant les compteurs géographiquement
#df_loc = df[['Nom_compteur','Latitude','Longitude','Comptage_horaire']]


#Création de la carte grâce au module Folium

#map = folium.Map(location=[df_loc.Latitude.mean(), df_loc.Longitude.mean()], zoom_start=12.5, control_scale=True)

#for index, data in df_loc.iterrows():
   # folium.Marker([data["Latitude"], data["Longitude"]], popup=data["Nom_compteur"]).add_to(map)



# call to render Folium map in Streamlit
# folium_static(map)

# #Ajout d'une colonne à df_loc pour séparer la variable comptage horaire en 4 plages auxquelles on attribuera différentes couleurs
# df_loc['Marker_color'] = pd.cut(df_loc['Comptage_horaire'], bins = 4, labels = ['dodgerblue','limegreen','yellow','orange'])

# map = folium.Map(location=[df_loc.Latitude.mean(), df_loc.Longitude.mean()], zoom_start=12.5, control_scale=True)

# #Création des cercles de couleur et de taille changeant fonction de la valeur du comptage horaire
# for index, data in df_loc.iterrows():
#     folium.Circle([data["Latitude"], data["Longitude"]], popup=data["Comptage_horaire"], radius = data['Comptage_horaire']*3, color = "w", fill_color = data['Marker_color'], fill_opacity = 0.4).add_to(map)

# for index, data in df_loc.loc[df_loc["Nom_compteur"] == "28 boulevard Diderot E-O"].iterrows():
#     folium.Marker([data["Latitude"], data["Longitude"]], popup=data["Comptage_horaire"], tooltip = "Compteur le moins déclenché").add_to(map)

# for index, data in df_loc.loc[df_loc["Nom_compteur"] == "Totem 73 boulevard de Sébastopol S-N"].iterrows():
#     folium.Marker([data["Latitude"], data["Longitude"]], popup=data["Comptage_horaire"], tooltip = "Compteur le plus déclenché").add_to(map) 
    
