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

## Ingénierie

- **Docker** : image CPU-only (wheels torch CPU explicites), `docker-compose.yml`
  avec service Ollama séparé.
- **CI** (GitHub Actions) : tests pytest + build Docker sur push/PR vers `main`.
- **MLflow** : tracking des runs d'entraînement (`mlruns/`).
- **Tests** (`tests/`) : mesures, métriques, synthèse chatbot.

## Limites connues

- Dataset réduit (68 images d'entraînement) : les performances sur des
  morphologies de plaie très différentes de celles vues à l'entraînement ne sont
  pas garanties.
- Biais de sur-segmentation observé sur le U-Net (recall > precision) : à
  surveiller si le cas d'usage exige une precision élevée plutôt qu'un bon
  compromis global.
- Performances dégradées sur les plaies quasi refermées (< ~3 % du champ) —
  voir analyse d'erreur ci-dessus.
- Le Hausdorff normalisé du U-Net n'a pas encore d'explication définitive (voir
  section Résultats) ; une investigation plus poussée (ex : Hausdorff calculé
  séparément sur chaque extrémité de la bande vs sur son corps) pourrait la
  confirmer.

## Reproduire les résultats

```bash
python -m src.split_data                                          # (re)génère les splits
python -m src.train --run_name unet_resnet34_holdout               # entraînement
python -m src.evaluate --run_name unet_resnet34_holdout --split test  # évaluation U-Net
python -m src.baseline --split test                                 # évaluation baseline
python -m src.error_analysis_figures                                 # figures du rapport
```

Résultats bruts : `outputs/predictions/test_metrics.json`,
`outputs/predictions/test_baseline_metrics.json`,
`outputs/predictions/test_comparisons/`, `outputs/predictions/test_baseline_comparisons/`,
`outputs/figures/`.
