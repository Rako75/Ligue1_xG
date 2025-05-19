📊 Analyse xG des Équipes de Ligue 1 (Top 9)
Ce projet Python permet de visualiser les performances offensives et défensives (via les xG) des 9 meilleures équipes de Ligue 1 sur les saisons 2022-2023, 2023-2024 et 2024-2025. Les visualisations affichent les tendances d'Expected Goals (pour et contre) selon une moyenne glissante de 10 matchs.

🖼️ Résultat
Le script génère une figure finale xG_Ligue1_Top9.png représentant :

Une courbe pour les xG pour et contre par équipe.

Une zone colorée avec un dégradé selon l'écart entre attaque et défense.

Le logo et le nom de l’équipe avec les dernières valeurs d'xG.

📁 Données nécessaires
Le script utilise 3 fichiers CSV :

Scores_and_Fixtures_Ligue_1_22_23.csv

Scores_and_Fixtures_Ligue_1_23_24.csv

Scores_and_Fixtures_Ligue_1_24_25.csv


📦 Installation
Installe les dépendances requises avec pip :

bash
Copier
Modifier
pip install highlight_text adjustText fuzzywuzzy
Autres bibliothèques nécessaires (souvent déjà présentes) :

bash
Copier
Modifier
pip install pandas numpy matplotlib seaborn beautifulsoup4 pillow
🚀 Exécution
Assure-toi que tous les fichiers CSV, les images (logos) et la police sont bien placés aux bons emplacements, puis exécute le script Python :

bash
Copier
Modifier
python script_xg_ligue1.py
L’image finale sera enregistrée sous le nom xG_Ligue1_Top9.png.

📌 Fonctions clés
get_xG_rolling_data(team_id, window): Calcule les moyennes glissantes des xG pour/contre.

get_xG_interpolated_df(team_id): Interpole les courbes pour un rendu plus fluide.

plot_xG_gradient(ax, team_id): Trace les courbes avec dégradé selon l’écart xG.

Visualisation multi-axes via matplotlib.gridspec.

📊 Visualisation
Fond bleu Ligue 1.

Titre et sous-titre avec mise en forme personnalisée.

Chaque équipe a son mini-graphe et une section texte/visuelle avec son logo.

Logos et texte enrichi via highlight_text.

📌 Auteur
Alex Rakotomalala

📚 Données
Données issues du site FBref, section Ligue 1.
