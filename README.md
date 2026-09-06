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
   Scholar), historique de conversations persistant.

### Base documentaire du RAG

14 articles sélectionnés manuellement (`chatbot/articles_cicatrisation.json`), et non
100 articles récupérés automatiquement sans tri : chaque candidat a été examiné (titre +
résumé) et retenu ou rejeté selon sa pertinence réelle pour le projet, avec la décision et
la raison tracées dans [`chatbot/curation_log.csv`](chatbot/curation_log.csv).
Chaque article retenu porte une `category` (voir `chatbot/index_articles.py`).

Couverture actuelle :
- ✅ Principes et limites du scratch assay (3)
- ✅ Automatisation et analyse d'image (10)
- ⚠️ Deep learning / segmentation biomédicale : **0 article** — les requêtes Semantic
  Scholar ciblées ont échoué (429 persistant sur l'API publique, sans clé, 3 tentatives).
  Voir Perspectives ci-dessous.
- ⚠️ Métriques d'évaluation (Dice/IoU/Hausdorff) : 1 seul article, pour la même raison —
  lacune jugée moins urgente que la précédente, ces métriques étant de toute façon
  calculées et validées directement par le code ([src/metrics.py](src/metrics.py)), pas
  par le LLM.
- ❌ Applications biologiques (traitements/molécules) retirées volontairement : elles
  parlent de cicatrisation via l'effet d'une molécule précise, pas du mécanisme de
  cicatrisation/migration ni de la méthode de mesure elle-même — hors du cœur du projet.

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

  Un seul run est tracé à ce jour (celui utilisé pour tous les résultats de ce
  README) ; le tableau est amené à s'étoffer si d'autres configurations sont
  testées.
- **Tests** (`tests/`) : mesures, métriques, synthèse chatbot.

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
- Le Hausdorff normalisé du U-Net est élevé (0,323) mais l'expérience TTA ci-dessus
  (section Résultats) confirme l'hypothèse : ce sont des artefacts de contour isolés
  (lissés par le TTA, sans changer le Dice), pas un problème de recouvrement global.

## Perspectives

- **Base documentaire du RAG** : les catégories deep learning/segmentation et métriques
  restent sous-couvertes (voir ci-dessus) parce que l'API publique Semantic Scholar est
  saturée sans clé (429 persistant sur 3 tentatives), pas par manque d'articles pertinents
  disponibles. Solution identifiée et prête à appliquer : une clé API Semantic Scholar
  gratuite lève cette limite de débit — `chatbot/fetch_articles_by_category.py` n'a besoin
  d'aucune autre modification pour en profiter. Non appliqué pour l'instant : c'est un
  enrichissement périphérique du RAG, pas une correction du cœur du projet (segmentation,
  évaluation, baseline), et le RAG reste fonctionnel et honnête sur cette limite en l'état.
- **Hyperparameter tuning** : un seul run MLflow tracé à ce jour (voir Ingénierie) ; tester
  d'autres encodeurs/`img_size`/learning rates permettrait de savoir si `unet_resnet34_holdout`
  est déjà un optimum local ou s'il reste de la marge.

## Reproduire les résultats

```bash
python -m src.split_data                                          # (re)génère les splits
python -m src.train --run_name unet_resnet34_holdout               # entraînement
python -m src.evaluate --run_name unet_resnet34_holdout --split test  # évaluation U-Net
python -m src.evaluate --run_name unet_resnet34_holdout --split test --tta  # idem + TTA (voir Résultats)
python -m src.baseline --split test                                 # évaluation baseline
python -m src.error_analysis_figures                                 # figures d'analyse d'erreur
python -m src.plot_training_curves --run_name unet_resnet34_holdout  # courbes d'apprentissage
```

Résultats bruts : `outputs/predictions/test_metrics.json`,
`outputs/predictions/test_metrics_tta.json`,
`outputs/predictions/test_baseline_metrics.json`,
`outputs/predictions/test_comparisons/`, `outputs/predictions/test_comparisons_tta/`,
`outputs/predictions/test_baseline_comparisons/`, `outputs/figures/`.
