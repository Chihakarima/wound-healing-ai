# Segmentation automatique de plaie (wound healing / scratch assay)

Pipeline de segmentation automatique de plaie par deep learning (U-Net + encodeur
ResNet34 pré-entraîné), avec une baseline classique de comparaison, une évaluation
rigoureuse sur un jeu de test tenu à l'écart, une application Streamlit pour
l'utilisation courante, et un assistant IA (RAG) pour interpréter les résultats
à la lumière de la littérature scientifique.

## Objectif

Remplacer la mesure manuelle de la surface de plaie (tracé au trait, long et
subjectif) par une mesure automatique reproductible, tout en évaluant sérieusement
si le gain en précision/temps justifie l'usage du deep learning par rapport à une
méthode classique de traitement d'image.

## Données

- 97 images de microscopie (scratch assay), chacune annotée manuellement par un
  contour tracé à la main sur la photo (`data/masques/`), pas directement un masque
  binaire exploitable.
- [src/prepare_masks.py](src/prepare_masks.py) reconstruit les masques binaires
  (`data/masks_binary/`) : détection du trait par couleur/teinte, pontage des trous
  (le trait s'efface par endroits à cause de la compression JPEG), fermeture sur les
  bords d'image, remplissage par flood-fill. Une image (`74`) est exclue : son tracé
  est incomplet même après ce traitement.
- [src/split_data.py](src/split_data.py) : split reproductible (seed 42) —
  **68 train / 14 val / 14 test**. Le split test est tenu à l'écart de
  l'entraînement ET de la sélection du checkpoint (contrairement à val), donc son
  score est une vraie estimation de généralisation.

## Méthode

### Modèle

U-Net avec encodeur ResNet34 pré-entraîné ImageNet
([segmentation_models_pytorch](src/model.py)). Fine-tuning en 2 phases (encodeur
gelé 5 époques puis dégelé à LR/10), loss Dice+BCE, `ReduceLROnPlateau`, early
stopping sur le Dice de validation, tracking MLflow. Voir [src/train.py](src/train.py).

**Courbes d'apprentissage** (`outputs/figures/figure3_loss_curves.png`,
`outputs/figures/figure4_dice_curves.png`, générées par
[src/plot_training_curves.py](src/plot_training_curves.py)) : train vs validation
uniquement — le jeu de test reste tenu à l'écart et n'est évalué qu'une seule fois
(voir Évaluation ci-dessous), pas suivi époque par époque. Les deux courbes
montrent une convergence régulière, sans signe de surapprentissage (le Dice de
validation ne décroche pas du Dice de train). Le Dice de validation est
au contraire *supérieur* à celui de train tout au long de l'entraînement : c'est
attendu ici, car l'augmentation de données forte appliquée à l'entraînement
(crop aléatoire, flips, bruit...) rend la tâche plus difficile à ce stade que la
validation, qui ne fait qu'un simple redimensionnement — ce n'est pas un
signal d'alerte. Le meilleur checkpoint (époque 54, Dice validation 0,923) est
sélectionné sur ce critère, conformément à l'early stopping utilisé.

### Baseline classique (comparaison)

Le gap n'est pas caractérisé par une couleur différente du reste du champ (même
rose des deux côtés), mais par l'**absence de texture cellulaire** (le gap est
lisse, les zones couvertes de cellules sont grainées). La baseline
([src/baseline.py](src/baseline.py)) seuille (Otsu) une carte d'écart-type local,
restreinte au champ du microscope (marge de vignettage exclue), suivie d'un
nettoyage morphologique — une approche comparable à celle des outils classiques de
mesure de scratch assay (ex : plugin ImageJ "Wound Healing Size Tool").

### Évaluation

Les deux méthodes sont évaluées **en résolution native** : le masque prédit (calculé
sur l'image redimensionnée à `img_size` pour le réseau, comme le voit le U-Net) est
réupsamplé à la résolution originale avant comparaison au masque de vérité terrain
original — exactement ce qu'un utilisateur voit dans l'application. Ça rend les deux
méthodes directement comparables sur les mêmes métriques, y compris la distance de
Hausdorff (voir plus bas). Voir [src/evaluate.py](src/evaluate.py).

## Résultats (14 images de test, jamais vues à l'entraînement)

| Méthode | Dice | IoU | Precision | Recall | Hausdorff normalisé* | MAE surface (px²) | Erreur de surface relative |
|---|---:|---:|---:|---:|---:|---:|---:|
| Texture + Otsu (baseline) | 0,576 | 0,448 | 0,558 | 0,817 | 0,204 | 210 243 | 404,9 % |
| **U-Net + ResNet34** | **0,882** | **0,796** | **0,852** | 0,929 | 0,323 | **8 538** | **15,9 %** |

\* Hausdorff / diagonale de l'image, pour rendre la distance comparable entre images
de résolutions différentes (0 = parfait, plus haut = pire).

**Lecture :** le U-Net domine largement sur toutes les métriques de recouvrement
(Dice, IoU, precision, recall) et surtout sur l'erreur de surface — le chiffre le
plus parlant pour un biologiste (**15,9 % d'erreur relative en moyenne contre
405 % pour la baseline**, qui sur-segmente massivement dès que les conditions
d'éclairage s'écartent du cas standard).

Fait notable et rapporté sans le masquer : le Hausdorff normalisé du U-Net
(0,323) est plus élevé (pire) que celui de la baseline (0,204), malgré un Dice
très supérieur. Le Hausdorff est une distance de pire cas (sensible à un seul
point de contour mal placé), et reste élevé même sur les meilleures images U-Net
(Dice > 0,9) — ce n'est donc probablement pas une vraie faiblesse du modèle mais
un effet de la géométrie de la plaie (bande fine et longue, dont les deux
extrémités sont les points les plus instables à localiser précisément pour les
deux méthodes). Recouvrement (Dice/IoU/precision/recall) et erreur de surface
restent les métriques les plus pertinentes pour l'objectif du projet.

**Piste testée pour ce Hausdorff élevé — Test-Time Augmentation (TTA)** : moyenner les
probabilités prédites sur 4 versions retournées de l'image (flips horizontal/vertical/180°,
exactement inversibles) avant seuillage
(`python -m src.evaluate --run_name unet_resnet34_holdout --split test --tta`, voir
[src/predict.py](src/predict.py)) fait chuter le Hausdorff normalisé de 0,323 à **0,043**
(Dice quasi inchangé : 0,882 → 0,886 ; erreur de surface légèrement dégradée : 15,9 % → 16,9 %).
Ça confirme l'hypothèse ci-dessus : le Hausdorff élevé venait d'artefacts de contour isolés aux
extrémités de la plaie, lissés par le TTA, pas d'un problème de recouvrement global. Gardé
**opt-in** (pas le comportement par défaut de l'app ni des figures ci-dessous) : ~4x plus lent par
image, et l'objectif du projet reste le recouvrement/la surface plutôt que le Hausdorff.

### Robustesse inter-seed

Le run canonique ci-dessus (`unet_resnet34_holdout`) n'était pas reproductible à la graine près
(aucune graine fixée dans une version antérieure de [src/train.py](src/train.py)). Pour vérifier
que ce résultat ne tient pas à un split ou une initialisation favorables, le même protocole
(mêmes hyperparamètres, split + initialisation régénérés à chaque fois, seed réellement fixée
via `--seed`) a été répété sur 3 graines (`unet_resnet34_seed42/123/2026`) :

| Seed | Dice | IoU | Erreur de surface | Hausdorff normalisé |
|---|---:|---:|---:|---:|
| 42 | 0,880 | 0,791 | 16,1 % | 0,083 |
| 123 | 0,904 | 0,828 | 7,2 % | 0,131 |
| 2026 | 0,892 | 0,809 | 8,8 % | 0,032 |
| **moyenne ± écart-type** | **0,892 ± 0,009** | **0,809 ± 0,015** | **10,7 % ± 3,9 %** | **0,082 ± 0,040** |

**Lecture :** le Dice est stable d'une graine à l'autre (écart-type de 0,009 sur une moyenne de
0,892, cohérent avec le 0,882 du run canonique) — la performance du modèle n'est pas un coup de
chance lié au split. Le Hausdorff normalisé, en revanche, varie énormément selon la graine (de
0,032 à 0,131, presque ×4) : ça confirme, avec des données cette fois, qu'il ne doit pas être
sur-interprété comme un score de qualité stable — sa valeur dépend fortement de détails
d'initialisation qui affectent surtout les extrémités de contour, pas du recouvrement global
(cohérent avec l'hypothèse et l'expérience TTA ci-dessus).

## Analyse d'erreur

**Figure 1** (`outputs/figures/figure1_exemples.png`) — quatre cas représentatifs
(image originale | vérité terrain | U-Net | baseline) :
- Bonne segmentation (id 39) : les trois méthodes suivent le contour, la baseline
  ajoute du bruit (petits blobs) mais reste dans la bonne zone.
- Cas difficile (id 50) : le U-Net reste précis, la baseline se fragmente.
- Plaie très petite (id 93) : voir ci-dessous.
- Échec de la baseline (id 76) : le U-Net suit correctement le contour, la
  baseline part en dérive et trace presque tout le pourtour du champ — l'image a
  un éclairage/contraste atypique (fond grisâtre plutôt que rose) qui rend le
  seuillage de texture inutilisable, illustrant la fragilité d'une méthode
  classique face à des conditions d'acquisition variables.

**Figure 2** (`outputs/figures/figure2_surface_vs_dice.png`) — surface relative de
la plaie (% du champ) vs Dice du U-Net sur les 14 images de test : tendance
observée où le Dice diminue pour les plaies de très petite surface relative.
Avec seulement 14 points, ce n'est pas une relation statistique démontrée, mais
un signal cohérent à garder en tête pour l'usage du modèle sur des plaies
quasi refermées.

**Cas de l'image 93** (pire score des deux méthodes : Dice U-Net 0,656→0,651 en
résolution native, baseline 0,459) : cette image a **la plus petite surface de
plaie de tout le dataset (97 images), rang 1/97, 1,3 % du champ** — plus petite
encore que l'image 74, explicitement exclue du dataset pour annotation
incomplète. Sur une structure aussi fine, quelques pixels d'écart de contour
suffisent à faire chuter fortement le Dice/IoU (métriques proportionnelles à la
surface). Ce n'est ni un bug ni un échec de généralisation isolé, mais une limite
connue et mesurable sur les plaies quasi refermées, partagée par les deux méthodes.

### Analyse d'erreur quantitative par catégorie

Pour vérifier avec des chiffres (pas seulement les 4 cas de la figure 1) les deux
hypothèses ci-dessus — dégradation sur les plaies quasi refermées et fragilité de
la baseline face à un éclairage atypique —
[src/error_analysis_categories.py](src/error_analysis_categories.py) croise les
métriques déjà calculées avec deux critères **objectifs et reproductibles** (pas
un classement visuel manuel) :
- surface relative de la plaie, seuillée à 3 % (le seuil déjà utilisé plus haut) ;
- luminosité moyenne du champ, comparée par z-score à la distribution du dataset
  complet (97 images) — un proxy de l'éclairage/contraste atypique évoqué pour
  l'image 76.

| Éclairage (test, n) | Dice U-Net | Dice baseline |
|---|---:|---:|
| Normal (12) | 0,884 | 0,656 |
| Atypique (2 : images 76, 91) | 0,874 | **0,092** |

**Lecture :** sur les 2 images de test à l'éclairage atypique (z-score de
luminosité \|z\| > 1,5 vs le dataset complet), le U-Net reste stable (Dice 0,874,
quasi identique au reste du test) alors que la baseline s'effondre (Dice moyen
0,092, erreur de surface > 1000 %) — ça confirme avec des données, au-delà du
seul cas 76 illustré en figure 1, que la fragilité de la baseline au contraste
est systématique et pas un exemple isolé choisi après coup.

**Avertissement sur la taille d'échantillon :** avec seulement 14 images de test,
chaque catégorie contient 1 à 13 images (voir
`outputs/predictions/test_error_category_summary.csv`, colonne `n`) — ce tableau
sert à vérifier une tendance déjà observée qualitativement, pas à établir un
résultat statistique. La catégorie "quasi fermée" ne contient qu'1 image (la 93,
déjà discutée ci-dessus) et n'est reportée qu'à titre indicatif.

## Application (Streamlit)

`streamlit run app.py` — trois onglets :
1. **Analyse d'une image** : masque, contour, surface (px²/cm²), suppression
   interactive de fragments de masque, prétraitement optionnel (débruitage,
   flat-field, CLAHE), correction manuelle du contour au pinceau, comparaison à
   un masque de référence (Dice, IoU, erreur de surface).
2. **Suivi de cicatrisation** : suivi temporel multi-images, % de fermeture,
   courbe, résumé scientifique généré par LLM.
3. **Assistant IA** : chatbot RAG (ChromaDB + Ollama/mistral) confrontant les
   résultats de segmentation à la littérature scientifique (articles Semantic
   Scholar), historique de conversations persistant. Réponse structurée en 4
   sections (📊 Résultats observés / 🔬 Interprétation / ⚠️ Limites / 📚
   Littérature, voir `PROMPT_TEMPLATE` dans [chatbot/chatbot.py](chatbot/chatbot.py))
   pour que le biologiste distingue toujours ce qui est mesuré de ce qui est
   interprété, avec les mêmes garde-fous que le résumé scientifique ci-dessous
   (ne jamais inventer une concordance avec la littérature, dire explicitement
   quand le corpus ne permet pas de conclure).

### Base documentaire du RAG

37 articles sélectionnés manuellement (`chatbot/articles_cicatrisation.json`), et non
100+ articles récupérés automatiquement sans tri : chaque candidat a été examiné (titre +
résumé officiel vérifié à la source) et retenu ou rejeté selon sa pertinence réelle pour le
projet, avec la décision et la raison tracées dans
[`chatbot/curation_log.csv`](chatbot/curation_log.csv). Chaque article retenu porte une
`category` (voir `chatbot/index_articles.py`).

Couverture actuelle :
- ✅ Principes, limites et protocole du scratch assay (6)
- ✅ Automatisation et analyse d'image (10)
- ✅ Quantification de la fermeture (4)
- ✅ Migration cellulaire et prolifération (4)
- ✅ Analyse d'image classique (seuillage, texture, outils historiques) (3)
- ✅ Deep learning / segmentation biomédicale (5) — U-Net original, un comparateur direct
  U-Net appliqué au scratch assay (Dice 0,958-0,968 sur 400+ images, à mettre en regard du
  Dice 0,882 du projet sur 97 images), et la justification de l'encodeur pré-entraîné.
- ✅ Métriques d'évaluation (Dice/IoU/Hausdorff) (3) — dont la référence qui explique
  l'instabilité du Hausdorff déjà documentée empiriquement dans ce projet (voir Résultats).
- ✅ Analyse temporelle / cinétique (2) — la plus difficile à couvrir : le candidat le plus
  pertinent était déjà dans le corpus, voir `curation_log.csv`.
- ⏸️ 2 articles écartés de ce tour d'ajout faute de résumé officiel accessible en texte
  intégral (Otsu 1979, paywall IEEE ; Seghier 2024, paywall Wiley) — non intégrés plutôt
  que résumés inventés, conformément à la règle de non-invention.
- ❌ Applications biologiques (traitements/molécules) retirées volontairement : elles
  parlent de cicatrisation via l'effet d'une molécule précise, pas du mécanisme de
  cicatrisation/migration ni de la méthode de mesure elle-même — hors du cœur du projet.

**Limite de récupération constatée après cet élargissement** : la recherche sémantique
(`chatbot/index_articles.py`, embeddings par défaut de ChromaDB, orientés anglais) retrouve
correctement les nouveaux articles sur une question formulée en anglais avec le bon
vocabulaire technique, mais pas toujours sur la même question posée en français par un
biologiste (ex : "Mon Dice de 0,882 est-il faible comparé à d'autres études U-Net sur
scratch assay ?" ne remonte pas l'article de comparaison directe pourtant présent dans le
corpus).

**Expérimentation contrôlée réalisée sur cette limite (résultat négatif informatif, pas
appliqué au projet)** : un embedding multilingue (`paraphrase-multilingual-MiniLM-L12-v2`
via `SentenceTransformerEmbeddingFunction`) a été testé sur un index ChromaDB temporaire, en
remplacement de l'embedding par défaut, sur les 37 articles du corpus. Changement de code
minimal (une ligne dans `index_articles.py`) et réindexation quasi instantanée (1,2 s pour 37
articles), mais coût de téléchargement initial du modèle d'environ 6-7 minutes (sans clé
Hugging Face). Résultat mesuré sur les deux questions test :
- Question sur la distance de Hausdorff : amélioration nette — "Metrics for evaluating 3D
  medical image segmentation" (Taha & Hanbury) et "Metrics reloaded" remontent désormais
  correctement, ce qui n'était pas le cas avec l'embedding par défaut.
- Question sur le Dice/U-Net : amélioration partielle seulement — l'architecture U-Net
  originale et TernausNet remontent, mais **pas** l'article de comparaison directe (Doğru et
  al. 2024), le plus pertinent pour cette question précise.

**Conclusion retenue, en toute rigueur expérimentale** : l'embedding multilingue apporte une
amélioration réelle mais partielle, pas une résolution complète du problème — donc pas un
correctif validé à intégrer tel quel. Changement non conservé : `sentence-transformers`
n'a pas été ajouté à `requirements.txt`, et l'index temporaire de test n'a pas été gardé.
Pistes pour une itération future (voir Perspectives) : un modèle multilingue plus grand
(`multilingual-e5-base`), ou une recherche hybride mots-clés + embeddings.

### Fiabilité de la génération par LLM

Plusieurs tests en direct du résumé scientifique (`generer_resume_stream`, onglet Suivi de
cicatrisation) ont révélé que mistral 7B (modèle local, sans garantie de suivi d'instruction)
pouvait confondre le taux d'un intervalle avec le taux moyen global, inventer un nombre de
mesures ne correspondant à aucune donnée fournie, ou une fois fabriquer des citations
complètes absentes des extraits fournis. Ces erreurs ont été corrigées non pas en renforçant
les consignes de la section où le LLM se trompait, mais en déplaçant chaque chiffre critique
vers la section où le LLM s'est montré fiable à la copie littérale ("Résultats observés") —
ou en le pré-calculant entièrement dans le code (`_phrase_heterogeneite`) pour qu'il n'ait
plus qu'à le recopier mot pour mot.

**Limite de robustesse du prompt (expérience isolée).** Une variante de prompt a ensuite été
testée pour améliorer la section "Mise en contexte scientifique" (expliquer ce que le corpus
couvre en général plutôt qu'un simple rejet). Cette variante, qui introduisait un patron
ouvert avec des champs à compléter librement (`[sujet général...]`, `[durée]h`), a provoqué
une dégradation de la fiabilité de **tout le résumé**, pas seulement de cette section : sur 3
générations consécutives, des surfaces fabriquées (87 646 px², 72 761 px² au lieu de
56 758 px²) et des bornes d'intervalle inexistantes (32h-48h, 0h-8h) sont apparues — alors que
cette nouvelle consigne, elle, ne s'est jamais mal exécutée. Un test d'isolation (même prompt,
seule cette consigne remise à sa version courte) a confirmé que la régression venait de cette
modification précise, pas d'un changement terminologique effectué en parallèle ni d'une
défaillance ponctuelle du modèle : 2/2 générations à nouveau correctes immédiatement après le
retour arrière. La variante a été retirée ; le prompt précédent a été conservé après
vérification des valeurs produites.

Cette expérience montre qu'avec le modèle local utilisé, l'ajout de consignes plus complexes
peut dégrader des parties du comportement précédemment fiables, pas seulement échouer sur son
propre objectif. Toute modification du prompt est donc désormais validée par plusieurs
générations réelles avant d'être conservée, jamais supposée correcte après une seule lecture
du texte produit. [tests/test_chatbot_llm_regression.py](tests/test_chatbot_llm_regression.py)
verrouille cette exigence : il génère un résumé réel (Ollama requis, ignoré automatiquement
sinon, y compris en CI) sur des mesures connues et vérifie que le texte contient les valeurs
calculées par le pipeline et ne contient aucune des valeurs fabriquées observées pendant le
développement.

## Ingénierie

- **Docker** : image CPU-only (wheels torch CPU explicites), `docker-compose.yml`
  avec service Ollama séparé.
- **CI** (GitHub Actions) : tests pytest + build Docker sur push/PR vers `main`. C'est
  de l'intégration continue (le code et l'image restent valides), pas du
  déploiement continu : aucune publication ni déploiement automatisé.
- **MLflow** : tracking des runs d'entraînement (backend SQLite `mlflow.db`,
  artefacts dans `mlruns/`) — hyperparamètres, métriques par époque, checkpoint et
  log CSV sauvegardés comme artefacts pour chaque run.

  | Run | Encodeur | img_size | Époques | Train/Val | Meilleur Dice (val) |
  |---|---|---:|---:|---|---:|
  | `unet_resnet34_holdout` | resnet34 | 384 | 60 | 68 / 14 | 0,923 |
  | `unet_resnet34_seed42` | resnet34 | 384 | 60 | 68 / 14 | 0,921 |
  | `unet_resnet34_seed123` | resnet34 | 384 | 60 | 68 / 14 | 0,923 |
  | `unet_resnet34_seed2026` | resnet34 | 384 | 58 (early stop) | 68 / 14 | 0,914 |

  Les 3 derniers runs sont l'étude multi-seed de robustesse (voir Résultats ci-dessus) ; même
  configuration que `unet_resnet34_holdout`, split et initialisation régénérés par seed. Pas
  encore d'autre configuration (encodeur/img_size/lr) testée — voir Perspectives.
- **Tests** (`tests/`) : mesures, métriques, synthèse chatbot, régression LLM (Ollama requis,
  auto-ignorée sinon).

### Ce qui n'est volontairement pas fait

Pas de versioning des données/checkpoints volumineux (`.gitignore` les exclut,
en attendant un outil dédié type DVC), pas de registre de modèles, pas de
déploiement automatisé. Cohérent avec l'échelle du projet — limites connues,
pas des manques à corriger dans l'immédiat.

## Limites connues

- Dataset réduit (68 images d'entraînement) : les performances sur des
  morphologies de plaie très différentes de celles vues à l'entraînement ne sont
  pas garanties.
- Biais de sur-segmentation observé sur le U-Net (recall > precision) : à
  surveiller si le cas d'usage exige une precision élevée plutôt qu'un bon
  compromis global.
- Performances dégradées sur les plaies quasi refermées (< ~3 % du champ) —
  voir analyse d'erreur ci-dessus.
- Le Hausdorff normalisé du U-Net est élevé (0,323 sur le run canonique) et surtout **très
  variable** : l'expérience TTA ci-dessus (section Résultats) l'explique par des artefacts de
  contour isolés (lissés par le TTA, sans changer le Dice), et l'étude multi-seed (section
  Robustesse inter-seed) le confirme avec des données indépendantes (de 0,032 à 0,131 selon la
  seed, presque ×4) — à ne jamais lire comme un score stable, contrairement au Dice/IoU (stables,
  0,892 ± 0,009 sur les mêmes 3 seeds).

## Perspectives

- **Qualité de récupération du RAG en français** : la base documentaire couvre maintenant
  toutes les catégories prévues (37 articles, voir ci-dessus), mais la recherche sémantique
  par défaut (embeddings ChromaDB orientés anglais) ne remonte pas toujours l'article le plus
  pertinent sur une question posée en français par un biologiste, alors qu'elle le fait sur
  la même question en anglais avec le bon vocabulaire technique. Un premier essai avec un
  embedding multilingue a déjà été mené (voir section Base documentaire du RAG ci-dessus) :
  amélioration partielle mais pas suffisante pour être adoptée telle quelle. Pistes pour une
  itération plus poussée : un modèle multilingue plus grand (`multilingual-e5-base`), une
  recherche hybride mots-clés + embeddings, ou l'augmentation de `n_results` — chacune à
  valider par les mêmes tests de récupération avant/après, pas par un changement à l'aveugle.
- **Hyperparameter tuning** : un seul run MLflow tracé à ce jour (voir Ingénierie) ; tester
  d'autres encodeurs/`img_size`/learning rates permettrait de savoir si `unet_resnet34_holdout`
  est déjà un optimum local ou s'il reste de la marge.
- **Comparaison architecturale** : comparer U-Net/ResNet34 à une autre architecture de
  segmentation légère permettrait de savoir si le gain observé face à la baseline vient
  du deep learning en général ou spécifiquement de cette configuration.

## Reproduire les résultats

```bash
python -m src.split_data                                          # (re)génère les splits
python -m src.train --run_name unet_resnet34_holdout               # entraînement
python -m src.evaluate --run_name unet_resnet34_holdout --split test  # évaluation U-Net
python -m src.evaluate --run_name unet_resnet34_holdout --split test --tta  # idem + TTA (voir Résultats)
python -m src.baseline --split test                                 # évaluation baseline
python -m src.error_analysis_figures                                 # figures d'analyse d'erreur
python -m src.error_analysis_categories                              # analyse d'erreur quantitative par catégorie
python -m src.plot_training_curves --run_name unet_resnet34_holdout  # courbes d'apprentissage
```

Résultats bruts : `outputs/predictions/test_metrics.json`,
`outputs/predictions/test_metrics_tta.json`,
`outputs/predictions/test_baseline_metrics.json`,
`outputs/predictions/test_error_by_category.csv`,
`outputs/predictions/test_error_category_summary.csv`,
`outputs/predictions/test_comparisons/`, `outputs/predictions/test_comparisons_tta/`,
`outputs/predictions/test_baseline_comparisons/`, `outputs/figures/`.
