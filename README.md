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

Concrètement : transformer automatiquement des images de scratch assay en mesures
quantitatives de fermeture de plaie, puis contextualiser ces résultats à l'aide d'un
LLM connecté à la littérature scientifique. Le LLM intervient uniquement en aval des
mesures calculées et ne réalise ni segmentation, ni mesure d'image, ni diagnostic.

```
Image de plaie
      ↓
Segmentation automatique
(U-Net + ResNet34)
      ↓
Quantification de la surface
(px² / cm²)
      ↓
Suivi temporel
(0 h / 24 h / 48 h / ...)
      ↓
Analyse de la cinétique de fermeture
      ↓
Interprétation scientifique
      ↓
LLM + RAG
(littérature scientifique)
```

## Portée et limites d'usage

Ce projet est un **outil de quantification d'image** (surface de plaie, % de
fermeture, cinétique) pour la recherche in vitro (scratch assay), pas un outil
d'aide au diagnostic médical ni un dispositif clinique. Le modèle est entraîné et
évalué sur des images de scratch assay d'un seul jeu de données/microscope (voir
Limites connues) ; il ne doit pas être utilisé pour évaluer une plaie humaine réelle
ni pour une décision de soin. L'assistant IA (RAG, voir plus bas) met un résultat de
segmentation en regard de la littérature scientifique sur la cicatrisation in vitro
et signale explicitement quand un rapprochement n'est pas justifié — il ne fournit
ni diagnostic ni recommandation thérapeutique, et ses réponses doivent être lues comme
un contexte scientifique, pas un avis médical.

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

### Ensemble de modèles + estimation de confiance par pixel

Les 4 modèles indépendants déjà entraînés (`unet_resnet34_holdout` + les 3 graines de l'étude de
robustesse ci-dessous) peuvent être combinés au lieu d'utiliser un seul modèle
([src/ensemble.py](src/ensemble.py)) : chaque modèle prédit séparément, la prédiction finale est
le vote majoritaire pixel par pixel, et le nombre de modèles en désaccord à chaque pixel donne une
carte de confiance directement exploitable par le biologiste (superposée en vert = accord total,
rouge = désaccord partiel dans l'app Streamlit, onglet "Analyse d'une image").

Mesuré sur les 14 images de test (`python -m src.evaluate_ensemble --split test`) :

| Métrique | Modèle unique (`unet_resnet34_holdout`) | **Ensemble (4 modèles)** |
|---|---:|---:|
| Dice | 0,882 | **0,893** |
| IoU | 0,796 | **0,812** |
| Precision | 0,852 | **0,881** |
| Recall | 0,929 | 0,915 |
| Hausdorff normalisé | 0,323 | **0,038** |
| Erreur de surface relative | 15,9 % | **12,2 %** |

**Lecture :** l'ensemble améliore Dice, IoU, precision et Hausdorff simultanément (un modèle
unique n'améliore jamais tout à la fois sans compromis, voir le TTA ci-dessus qui gagne sur
Hausdorff mais perd légèrement sur l'erreur de surface) — cohérent avec le principe du vote
majoritaire, qui filtre les erreurs propres à un seul modèle sans propager ses faux positifs
isolés. Le Hausdorff normalisé (0,038) est même meilleur que celui obtenu par TTA (0,043),
sans le compromis sur l'erreur de surface. La légère baisse de recall (0,929 → 0,915) est le
compromis attendu : le vote majoritaire filtre aussi quelques vrais positifs marginaux prédits
par un seul modèle.

**La carte de confiance est un signal réel, pas un gadget visuel** : corrélation de **-0,751**
entre le % de zone en désaccord et le Dice par image sur le jeu de test — les images où les
modèles divergent le plus sont statistiquement celles où l'ensemble se trompe le plus. Le
désaccord se concentre visuellement aux extrémités fines de la plaie (vérifié visuellement dans
l'app), cohérent avec l'explication déjà donnée pour l'instabilité du Hausdorff (TTA et
robustesse inter-seed ci-dessus/ci-dessous).

**Vérification préalable** : les 4 modèles combinés ont été contrôlés individuellement sur leurs
courbes d'entraînement (`outputs/*_training_log.csv`) avant d'être combinés — val_loss < train_loss
et val_dice > train_dice jusqu'à la dernière époque pour les 4, aucun signe de surapprentissage
(cohérent avec l'explication déjà donnée pour `unet_resnet34_holdout` dans "Courbes
d'apprentissage" ci-dessus : augmentation forte au train, pas un signal d'alerte).

**Coût** : 4 modèles chargés et exécutés au lieu d'1 — plus lent et plus gourmand en mémoire,
option désactivée par défaut dans l'app (case à cocher dans la barre latérale).

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

### Significativité statistique (U-Net vs baseline)

Le tableau de résultats ci-dessus montre un écart de moyennes important (Dice 0,882 vs 0,576),
mais une moyenne seule ne dit pas si cet écart pourrait s'expliquer par le hasard
d'échantillonnage sur seulement 14 images. [src/statistical_comparison.py](src/statistical_comparison.py)
complète les métriques déjà calculées par un test des rangs signés de Wilcoxon (non paramétrique,
adapté à un petit échantillon) sur les paires (U-Net, baseline) de chaque image de test, et un
intervalle de confiance bootstrap sur la différence moyenne :

| Métrique | Moyenne U-Net | Moyenne baseline | Diff. moyenne | U-Net meilleur sur | p (Wilcoxon) | IC bootstrap 95 % de la diff. |
|---|---:|---:|---:|---:|---:|---:|
| Dice | 0,882 | 0,576 | +0,307 | 14/14 images | 0,000122 | [+0,183 ; +0,448] |
| IoU | 0,796 | 0,448 | +0,348 | 14/14 images | 0,000122 | [+0,236 ; +0,470] |

**Lecture :** le U-Net obtient un Dice et un IoU supérieurs sur les 14 images de test, sans
exception — c'est la configuration la plus favorable possible pour un test des rangs signés à
n=14, d'où le p le plus petit que ce test puisse produire à cette taille d'échantillon (0,000122,
donc largement < 0,05). L'écart n'est donc pas un artefact d'échantillonnage sur ces 14 images :
la supériorité du U-Net est statistiquement significative sur ce jeu de test, avec la réserve
habituelle qu'un échantillon de 14 images reste petit pour l'intervalle de confiance
(voir Dataset assez petit dans Limites connues) — un p très faible confirme la cohérence de
l'effet sur cet échantillon, pas sa généralisation à d'autres données.

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

`streamlit run app.py` — deux onglets :
1. **Analyse d'une image** : masque, contour, surface (px²/cm²), suppression
   interactive de fragments de masque, prétraitement optionnel (débruitage,
   flat-field, CLAHE), correction manuelle du contour au pinceau, comparaison à
   un masque de référence (Dice, IoU, erreur de surface). Option **estimation de
   confiance par ensemble de 4 modèles** (voir "Ensemble de modèles" ci-dessus) :
   carte verte/rouge superposée + alerte automatique si la zone de désaccord entre
   modèles dépasse 25 % de la zone détectée.
2. **Suivi de cicatrisation** : suivi temporel multi-images, % de fermeture,
   courbe, résumé scientifique généré par LLM, puis un **chat RAG intégré**
   (ChromaDB + Ollama/mistral) sous le résumé pour poser des questions libres
   sur ce même résultat — plus de contexte à ressaisir, l'assistant reprend
   automatiquement les mesures déjà calculées ci-dessus. Réponse structurée en
   4 sections (📊 Résultats observés / 🔬 Interprétation / ⚠️ Limites / 📚
   Littérature, voir `PROMPT_TEMPLATE` dans [chatbot/chatbot.py](chatbot/chatbot.py)),
   avec les mêmes garde-fous que le résumé scientifique ci-dessus (ne jamais
   inventer une concordance avec la littérature, dire explicitement quand le
   corpus ne permet pas de conclure). Chat simple, sans historique
   multi-conversations : une conversation par analyse, non persistée d'une
   session à l'autre (l'ancien onglet "Assistant IA" séparé, avec historique
   de conversations sur disque, a été retiré — redondant avec ce chat une
   fois le résultat déjà disponible dans le même onglet).

### Base documentaire du RAG

41 articles sélectionnés manuellement (`chatbot/articles_cicatrisation.json`), et non
100+ articles récupérés automatiquement sans tri : chaque candidat a été examiné (titre +
résumé officiel vérifié à la source) et retenu ou rejeté selon sa pertinence réelle pour le
projet, avec la décision et la raison tracées dans
[`chatbot/curation_log.csv`](chatbot/curation_log.csv). Chaque article retenu porte une
`category` (voir `chatbot/index_articles.py`).

Couverture actuelle :
- ✅ Principes, limites et protocole du scratch assay (7) — dont un dispositif open-source
  robotisé qui standardise la technique manuelle du scratch (source d'irreproductibilité
  connue), ajouté sur demande explicite pour couvrir le protocole de base utile à un
  biologiste débutant
- ✅ Automatisation et analyse d'image (10)
- ✅ Quantification de la fermeture (4)
- ✅ Migration cellulaire et prolifération (4)
- ✅ Analyse d'image classique (seuillage, texture, outils historiques) (3)
- ✅ Deep learning / segmentation biomédicale (6) — U-Net original, un comparateur direct
  U-Net appliqué au scratch assay (Dice 0,958-0,968 sur 400+ images, à mettre en regard du
  Dice 0,882 du projet sur 97 images), et la justification de l'encodeur pré-entraîné. Ajouté
  depuis (session 2026-09-08, ciblant l'échec persistant de la question de benchmark
  « mon Dice de 0,88 est-il faible ? », voir plus bas) : un second comparateur indépendant,
  U-Net + encodeur pré-entraîné ImageNet sur des micrographies de cellules souches
  mésenchymateuses (pas un scratch assay, mais même modalité — microscopie en contraste de
  phase — et même méthode), Dice = 0,876, quasi identique au Dice = 0,882 du projet. **Résultat
  négatif honnête** : cet ajout ne corrige pas la question de benchmark visée (toujours absente
  du top 10, voir plus bas) — confirme que le problème est la formulation/l'espace des
  embeddings pour cette question précise, pas un manque de contenu pertinent dans le corpus.
- ✅ Métriques d'évaluation (Dice/IoU/Hausdorff) (4) — dont la référence qui explique
  l'instabilité du Hausdorff déjà documentée empiriquement dans ce projet (voir Résultats), et,
  ajouté depuis (session 2026-09-08, gap identifié : aucun article ne reliait cette instabilité à
  la géométrie de la plaie), l'article introduisant clDice, une mesure/perte topologique conçue
  pour les structures tubulaires/fines — géométrie analogue à la bande fine et longue de la
  plaie, dont les extrémités sont déjà identifiées comme la source du Hausdorff instable (voir
  Résultats et Robustesse inter-seed). Candidat déjà repéré lors d'une curation antérieure mais
  laissé de côté comme « hors domaine » (résumé officiel vérifié sur arXiv, voir
  `curation_log.csv`) ; l'ajout est validé par une nouvelle question de benchmark dédiée
  (`chatbot/evaluate_retrieval.py`) qui le retrouve au rang 3 (rang 2 en recherche hybride).
  Benchmark final après les 3 ajouts de cette session (voir aussi ci-dessous) : Recall@3
  60 % (6/10) → **66,7 % (8/12)**, MRR@10 0,514 → **0,526**, sans aucune régression sur les 10
  questions d'origine (rangs tous inchangés, voir `retrieval_eval_results.json`).
- ✅ Analyse temporelle / cinétique (3) — la plus difficile à couvrir : le candidat le plus
  pertinent était déjà dans le corpus, voir `curation_log.csv`. Confirmé par une seconde
  recherche ciblée (session 2026-09-08, motivée par ce même constat) : sur 9 candidats
  récupérés, aucun n'apportait de contenu réellement nouveau et pertinent (doublons avec des
  articles déjà retenus, ou études d'intervention spécifique — ex: thérapie par pression
  négative — hors du cœur méthode/quantification du projet). Rien ajouté à cette catégorie à ce
  moment-là plutôt que d'y forcer un article marginal. Un article a toutefois été ajouté ensuite
  dans la même session : *Modeling epithelial wound closure dynamics with AI* (2025), qui
  combine segmentation (U-Net++ vs seuillage Otsu — parallèle direct avec la comparaison U-Net
  vs baseline Otsu de ce projet) et modélisation temporelle de la trajectoire de fermeture
  (Random Forest, ARIMA, réseau convolutif temporel). Résumé officiel vérifié sur PMC
  (PMC12528918). Validé par une question de benchmark dédiée (rang 1, voir plus bas) — mais ne
  répond pas à la question « combien de points de mesure sont nécessaires », qui reste sans
  réponse retrouvée (voir juste en dessous) : un ajout de contenu complémentaire, pas un correctif
  du trou de retrieval déjà identifié sur cette question précise.
  **Retrieval, pas contenu** : constaté en usage réel que la requête fixe utilisée par
  `generer_resume_stream` pour cette section ne retrouvait aucun des 2 articles de cette
  catégorie dans le top 10 (les 2 sont bien dans le corpus, juste jamais remontés pour cette
  formulation précise) — reformulée depuis, voir "Fiabilité de la génération par LLM" plus bas.
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

**Évaluation reproductible de la récupération** :
[chatbot/evaluate_retrieval.py](chatbot/evaluate_retrieval.py) transforme cette observation
ponctuelle (2 questions) en un petit benchmark reproductible de 10 questions représentatives,
chacune associée au(x) titre(s) d'article jugé(s) pertinent(s) par relecture manuelle (comme le
reste de la curation du corpus). Pour chaque question : rang du premier article attendu dans les
résultats (`None` si absent du top 10), et si l'article ressort dans le top 3 (`n_articles=3`,
la profondeur utilisée en production par `chatbot.py`). Résultat sur le corpus actuel (10
questions d'origine + 2 questions ajoutées lors de la session 2026-09-08 pour valider chacune un
nouvel article, voir ci-dessus) :

- **Recall@3 : 66,7 % (8/12)** — **MRR@10 : 0,526** (10 questions d'origine seules, pour
  référence historique : 60 % (6/10), MRR@10 0,514 — inchangé sur ces 10 questions précises,
  voir `retrieval_eval_results.json`)
- La question de comparaison directe Dice/U-Net échoue **dans les deux langues** avec cette
  formulation (l'article "An automated in vitro wound healing microscopy image analysis approach
  utilizing U-net-based deep learning methodology" n'apparaît dans le top 10 ni en anglais ni en
  français) : ça nuance l'anecdote ci-dessus — la limite n'est pas seulement un écart de langue,
  elle dépend aussi fortement du vocabulaire et de la formulation précise de la question, y
  compris en anglais.
- La question sur les limites du scratch assay retrouve l'article attendu au rang 7 : présent
  dans le corpus et retrouvable avec une recherche plus large, mais hors du top 3 utilisé par
  défaut par le chatbot.
- La question de cinétique (analyse temporelle) échoue entièrement, cohérent avec le fait déjà
  noté ci-dessus que cette catégorie est la plus difficile à couvrir.

Résultat détaillé par question : `chatbot/retrieval_eval_results.json` (régénéré par
`python -m chatbot.evaluate_retrieval`). Sur un si petit nombre de questions, ces chiffres servent
à objectiver une tendance déjà repérée qualitativement, pas à certifier un taux de succès général
du retrieval.

**Expérience de reformulation (la formulation, pas seulement la langue, est le vrai levier)** :
pour les 4 questions qui échouaient ci-dessus, `chatbot/evaluate_retrieval.py` teste une
reformulation à intention identique mais au vocabulaire plus technique/biomédical (repris des
résumés d'articles ciblés, ex : "Dice similarity coefficient" plutôt que "Dice score"), dans la
même langue que l'originale, pour isoler l'effet du vocabulaire de celui de la langue :

| Question | Rang original | Rang reformulé | Dans le top 3 ? |
|---|---:|---:|---|
| Comparaison Dice/U-Net (EN) | absent (> 10) | **1** | ✅ (était ❌) |
| Comparaison Dice/U-Net (FR) | absent (> 10) | 8 | ❌ (toujours hors top 3) |
| Limites du scratch assay (EN) | 7 | **2** | ✅ (était ❌) |
| Cinétique / points de mesure (FR) | absent (> 10) | 4 | ❌ (proche, toujours hors top 3) |

**Recall@3 sur ces 4 questions : 0 % → 50 % après reformulation.** Deux cas s'améliorent nettement
avec une formulation plus technique en gardant la même langue (le levier n'est donc pas seulement
la traduction) ; les deux autres (les deux en français) progressent aussi (absent du top 10 →
rang 8 et 4) mais restent hors du top 3 utilisé en production — cohérent avec l'hypothèse d'un
écart de vocabulaire qui se cumule ici avec un écart de langue, sans qu'un seul des deux facteurs
suffise à l'expliquer entièrement. Résultat : la reformulation ciblée est une piste réelle et peu
coûteuse (aucun changement de code, juste la question posée), mais ne résout pas seule le cas
français le plus difficile — cohérent avec les pistes déjà notées en Perspectives (embedding
multilingue, recherche hybride).

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

**Recherche hybride (mots-clés + embeddings) : gain mesuré sur le retrieval, non retenu pour
la génération après reconsidération.** `chatbot/index_articles.py` propose
`chercher(question, n, hybride=True)` : fusionne la recherche par embeddings (ChromaDB) et une
recherche par mots-clés (BM25 via `rank_bm25`, gratuit et local, ajouté à `requirements.txt`)
par reciprocal rank fusion. Changement de code minimal, aucun coût ni dépendance lourde.

- **Sur le benchmark de retrieval** (`chatbot/evaluate_retrieval.py`, qui compare les deux
  modes) : Recall@3 60 % → **70 %**, MRR@10 0,514 → **0,637**, sans aucune régression sur les
  10 questions — un gain net et mesuré.
- **Risque mesuré sur la génération** (`chatbot.py`) : activer `hybride=True` change quels
  articles réels atterrissent dans le contexte fourni au LLM pour la section "Mise en contexte
  scientifique" — sans toucher un seul mot du prompt lui-même. Un test d'isolation (même
  protocole que pour les modifications de prompt, voir section suivante) a montré que ce
  changement de contexte déstabilise quand même les sections numériques du résumé : 3/3
  générations en hybride ont fabriqué un taux de fermeture erroné et des bornes d'intervalle
  inexistantes (ex: "36-48h", alors que les mesures ne couvrent que 0/24/48h), contre 2/3
  correctes en mode original (embeddings seuls).
- **Décision finale** : brièvement activé en production malgré ce risque, sur demande
  explicite, puis désactivé à nouveau après reconsidération — pour un usage scientifique
  sérieux (chiffres destinés à être réutilisés tels quels), le risque de fabrication l'emporte
  sur le gain de retrieval. `hybride=False` dans les deux points d'appel de `chatbot.py` ;
  `hybride=True` reste disponible et validé pour le retrieval seul (`evaluate_retrieval.py`).
  `tests/test_chatbot_llm_regression.py` a été étendu avec les valeurs fabriquées observées
  pendant l'épisode hybride, au cas où ce mode d'échec réapparaîtrait autrement.
- **Leçon retenue** : avec ce modèle local, la fragilité déjà documentée ci-dessous ne se
  limite pas au texte du prompt — changer le contexte injecté (même avec du contenu réel,
  sans rien inventer) peut déstabiliser des sections du résumé qui n'utilisent pourtant pas ce
  contexte.

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
du texte produit.

**La fragilité ne se limite pas au texte du prompt.** Une troisième expérience (voir
"Recherche hybride" ci-dessus) a confirmé que changer uniquement le *contexte retrouvé* (des
articles réels différents, aucun mot du prompt modifié) suffit à déclencher la même
instabilité : 3/3 générations ont fabriqué des chiffres dans des sections qui n'utilisent
même pas ce contexte, contre 2/3 correctes avec le contexte d'origine. Ce risque a été activé
un temps en production sur demande explicite, puis désactivé à nouveau après reconsidération
(voir "Recherche hybride" ci-dessus) — la règle de validation avant activation tient toujours,
y compris pour un changement de contexte de retrieval, pas seulement pour le texte du prompt.
[tests/test_chatbot_llm_regression.py](tests/test_chatbot_llm_regression.py)
verrouille cette exigence : il génère un résumé réel (Ollama requis, ignoré automatiquement
sinon, y compris en CI) sur des mesures connues et vérifie que le texte contient les valeurs
calculées par le pipeline et ne contient aucune des valeurs fabriquées observées pendant le
développement.

**Un retrieval mal ciblé peut aussi ressembler à un problème de fiabilité, sans en être un.**
Signalé en usage réel : la section "Mise en contexte scientifique" du résumé de cinétique
répondait systématiquement qu'elle ne pouvait rien dire de pertinent. Diagnostic : la requête
fixe utilisée pour cette section ne retrouvait aucun des 2 articles `analyse_temporelle` du
corpus dans le top 10 — un vrai trou de retrieval (cohérent avec le Recall@3 déjà mesuré comme
faible sur cette catégorie), pas une mauvaise réponse du LLM face à un contexte correct. La
requête a été reformulée (vocabulaire repris des résumés d'articles ciblés, toujours
`hybride=False`), validée par le même test d'isolation (3/3 générations propres) avant d'être
gardée : `Sources` cite désormais des articles réellement pertinents (ex : "Study of Wound
Healing Dynamics by Single Pseudo-Particle Tracking..."). Gain **partiel** : le prompt continue,
à raison, à exiger une correspondance quantitative directe avant d'affirmer une concordance, donc
la prose de cette section reste souvent prudente même avec de meilleures sources en contexte.

**Retrieval encore meilleur ≠ génération plus fiable.** Une deuxième reformulation, ciblant en
plus l'article "The Frequent Sampling of Wound Scratch Assay..." (les 2 articles les plus
pertinents remontaient alors aux rangs 1 et 2, objectivement mieux que la reformulation
retenue), a été testée puis rejetée : 3/3 générations ont cette fois fabriqué la vitesse
moyenne et les vitesses par intervalle (ex : "7,2%/h" puis "0%/h" au lieu de 0,83%/h et
1,58%/h). Cause probable : le résumé de cet article est lui-même dense en chiffres de
cinétique (fenêtre de 6h, doses de médicaments, % de diminution de vitesse), que le modèle
semble mélanger aux données de l'utilisateur. Retenu : le score de retrieval d'un changement
ne dit rien de sa sécurité pour la génération — seul le test d'isolation en 3+3 générations
tranche.

**Même une consigne anti-fabrication peut elle-même provoquer une fabrication.** Suite à la
découverte ci-dessus, une phrase a été ajoutée à `PROMPT_RAPPORT` juste après les extraits
d'articles, avertissant explicitement le LLM que les chiffres des extraits appartiennent à
D'AUTRES expériences et ne doivent jamais être recopiés comme des mesures de l'utilisateur —
une clarification de bon sens, motivée par l'échec précédent. Testée en isolation (3
générations) : **3/3 échecs**, l'un d'eux pire que tout ce qui avait été observé jusque-là
(surface finale fabriquée à 5840 px² et fermeture à 93,76 % au lieu des vraies 56758 px² /
39,8 %, plus des bornes d'intervalle inventées comme "(48-24) heures"). Retirée immédiatement.
Constat : `PROMPT_RAPPORT` est désormais considéré comme étant à — ou au-delà de — son plafond
de complexité sûre pour ce modèle local ; toute nouvelle demande d'ajouter "juste une consigne
de plus" doit être accueillie avec scepticisme par défaut, quel que soit son bon sens apparent,
et testée avant toute autre considération.

**Une citation fabriquée, détectée en usage réel — corrigée sans toucher au prompt.** Un
résumé généré en conditions réelles a cité un article intitulé *"Analysis of wound healing
using digital image analysis"* : ce titre n'existe pas dans les 38 articles du corpus et ne
correspond à aucune des sources réellement retrouvées pour cette génération — un titre inventé,
pas un chiffre inventé, une variante que `test_chatbot_llm_regression.py` ne vérifiait pas.
Plutôt que de retenter une consigne de prompt (risque déjà démontré ci-dessus),
`chatbot/chatbot.py` gagne `detecter_citations_suspectes(texte, sources)` : une vérification
**après coup**, en code pur (regex + comparaison floue avec `difflib`), qui repère les
citations entre guillemets absentes des sources réellement retournées. Comme elle ne touche ni
au prompt ni au contexte retrouvé, elle échappe à la règle d'isolation ci-dessus (rien à
déstabiliser, c'est une lecture du texte déjà généré) — validée par
[tests/test_chatbot_citations.py](tests/test_chatbot_citations.py) (6 cas, dont le titre
fabriqué exact observé). Câblée dans `app.py` : un avertissement s'affiche sous les sources
chaque fois qu'une citation ne correspond à aucune d'elles, pour le résumé de cinétique comme
pour le chat. Ce n'est qu'un **détecteur** — il rend le problème visible, il ne l'empêche pas.

**"Mise en contexte scientifique" trop vague — séparée en un second appel LLM isolé, pas
renforcée dans le même prompt.** Signalé en usage réel (résumés générés manuellement) : cette
section se limitait souvent à mentionner que des extraits existaient, sans dire ce qu'ils
montrent réellement. Une première tentative de correction a suivi la méthode habituelle
(renforcer la consigne de cette section dans `PROMPT_RAPPORT`, sans template à trous cette
fois) et a été **testée en isolation avant/après réécriture** (3 générations de référence, 3
après modification, mêmes mesures) : la nouvelle consigne a bien amélioré cette section
(contenu réellement résumé, conclusion "quantification de cette série, pas une cinétique
universelle" comme voulu), mais a de nouveau déstabilisé des sections numériques qu'elle ne
touchait pourtant pas — vitesse moyenne fabriquée à 12,76 %/h, 4,2 %/h puis 7,2 %/h selon la
génération (vraie valeur : 0,83 %/h), bornes d'intervalle absurdes (`[100%-50%]`). 3/3 échecs :
modification abandonnée immédiatement, cohérent avec l'épisode "recherche hybride" ci-dessus
(le simple fait de renforcer ou changer le contenu autour des mesures suffit à les déstabiliser,
même sans toucher au texte qui décrit ces mesures).

Plutôt que de retenter une variante de prompt, `generer_resume_stream` a été restructurée en
**deux appels LLM indépendants** : `PROMPT_RAPPORT` ne contient plus du tout d'extraits
d'articles (3 sections : Résultats observés / Analyse quantitative / Interprétation prudente),
et un nouveau `PROMPT_CONTEXTE` séparé ne reçoit, à l'inverse, **aucune mesure** — seulement les
extraits — pour rédiger la 4ᵉ section. Les deux flux sont concaténés en Python après coup. La
contamination croisée devient ainsi structurellement impossible (le second appel n'a physiquement
pas accès aux chiffres pour les fabriquer), plutôt qu'évitée par une consigne qu'un modèle local
peut ne pas suivre. Testée en isolation (3 générations) : **0 fabrication numérique sur les 3
sections chiffrées**, et une "Mise en contexte scientifique" nettement plus substantielle dans
les 3 cas (contenu réel de chaque article résumé, conclusion attendue présente). Confirmé par
[tests/test_chatbot_llm_regression.py](tests/test_chatbot_llm_regression.py) (inchangé, toujours
vert) et l'ensemble de la suite de tests. Coût : deux appels séquentiels au lieu d'un, donc un
résumé complet prend plus longtemps à s'afficher en entier (chaque section reste toutefois
diffusée en streaming dès qu'elle est prête).

**Répétition du même caveat dans "Interprétation prudente" — retirée en simplifiant le prompt,
pas en le renforçant.** Signalé en usage réel : cette section répétait deux fois la même idée
("les vitesses par intervalle ne comparent que des taux moyens... ça ne démontre ni ne mesure une
accélération" en ouverture, puis "cette différence décrit uniquement les mesures disponibles et
ne permet pas... de conclure à une accélération biologique" dans la phrase-patron obligatoire
juste après) — deux phrases différentes dans leur formulation mais portant exactement le même
message. Confirmé systématique en relisant les transcripts d'isolation déjà produits pour les
corrections précédentes (présent dans la quasi-totalité des générations, avant comme après
l'architecture à deux appels ci-dessus). Cause identifiée dans le prompt lui-même : la consigne
demandait explicitement d'expliquer ce caveat en préambule, immédiatement suivie d'une
phrase-patron obligatoire qui l'exprime déjà, numériquement. Une détection de quasi-doublon par
similarité textuelle (comme `detecter_citations_suspectes`) ne suffirait pas ici : les deux
phrases ne se ressemblent pas assez littéralement (sujets différents) pour qu'une comparaison de
chaînes les repère de façon fiable — la redondance est conceptuelle, pas textuelle. Correction
retenue : suppression de la phrase de préambule redondante dans `PROMPT_RAPPORT` (la
phrase-patron, elle, reste seule et suffisante). Contrairement à tous les épisodes précédents de
cette section, il s'agit d'un **retrait** d'instruction, pas d'un ajout — testée en isolation
quand même (3 générations) : répétition absente dans les 3, toujours 0 fabrication numérique,
confirmé par la suite de tests complète (31/31).

**Couverture de test élargie à plusieurs scénarios de mesures, pas un seul fixture fixe.**
Limite identifiée lors d'une relecture du projet : jusque-là, `test_chatbot_llm_regression.py`
ne vérifiait qu'un seul jeu de mesures (3 points, ~94 000 px²) — un "3/3 propre" dessus ne
garantit rien sur un nombre de points différent ou un tout autre ordre de grandeur. Le test
couvre maintenant 3 scénarios distincts : le cas canonique (3 points), un cas limite structurel
(2 points, un seul intervalle — jamais couvert par un vrai appel LLM avant cet ajout), et un cas
à 5 points/grandes surfaces avec une coïncidence délibérée (la vitesse moyenne globale égale
numériquement le taux du dernier intervalle, pour stress-tester la confusion déjà observée entre
les deux). Les deux nouveaux scénarios ont d'abord "échoué" à cause d'un bug du test lui-même,
pas d'une fabrication : le LLM écrit parfois les grandes surfaces avec un séparateur de milliers
("50 000" au lieu de "50000"), une mise en forme différente du même nombre correct — corrigé en
acceptant les deux variantes plutôt qu'en durcissant le prompt. Aucune modification de prompt
dans ce changement (uniquement du code de test) : pas soumis à la règle d'isolation ci-dessus.
Suite complète : 33/33.

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
- **Tests** (`tests/`) : mesures, métriques, ensemble (vote majoritaire, carte de confiance),
  retrieval, synthèse chatbot, détection de citations fabriquées, régression LLM (Ollama requis,
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
- Pas de validation externe : les métriques ci-dessus sont mesurées sur un jeu de test tenu à
  l'écart mais issu du même dataset (même microscope, mêmes conditions d'acquisition). Aucune
  évaluation n'a été faite sur des images d'un autre laboratoire/microscope — la généralisation à
  d'autres conditions d'acquisition n'est donc pas démontrée, seulement plausible au vu de la
  robustesse inter-seed ci-dessus.

## Perspectives

- **Qualité de récupération du RAG** : la base documentaire couvre maintenant toutes les
  catégories prévues (41 articles, voir ci-dessus), mais la recherche sémantique par défaut
  (embeddings ChromaDB orientés anglais) ne remonte pas toujours l'article le plus pertinent. Le
  benchmark reproductible ci-dessus (`chatbot/evaluate_retrieval.py`, Recall@3 = 66,7 % (8/12),
  MRR@10 = 0,526) affine le diagnostic initial : ce n'est pas seulement un écart français/anglais (la
  question de comparaison Dice/U-Net échoue dans les deux langues avec cette formulation), mais
  une sensibilité plus large à la formulation et au vocabulaire de la question — confirmé une
  troisième fois (session 2026-09-08) : l'ajout d'un second comparateur Dice/U-Net directement
  pertinent au corpus (voir Base documentaire du RAG ci-dessus, Dice = 0,876) n'a **pas** suffi à
  faire remonter cette question dans le top 10, alors que le contenu répondrait pourtant bien à
  la question posée — la limite est donc bien dans la recherche (formulation/espace des
  embeddings), pas dans la couverture du corpus, contrairement à ce qu'un simple ajout d'articles
  pourrait laisser espérer. Un premier essai
  avec un embedding multilingue a déjà été mené (voir section Base documentaire du RAG ci-dessus) :
  amélioration partielle mais pas suffisante pour être adoptée telle quelle. Une recherche
  hybride mots-clés + embeddings a aussi été testée : gain net sur ce même benchmark
  (Recall@3 60 %→70 %, voir section "Recherche hybride" ci-dessus) mais déstabilise la
  génération du résumé LLM en aval — validée côté retrieval, pas encore exploitable côté
  chatbot sans repenser comment isoler la génération de ce contexte (ex : un appel LLM séparé
  par section, piste déjà notée plus haut). Pistes restantes pour une itération plus poussée :
  un modèle multilingue plus grand (`multilingual-e5-base`), l'augmentation de `n_results`, ou
  résoudre l'incompatibilité hybride/génération elle-même — chacune à valider par le même
  benchmark de retrieval ET par le test d'isolation de génération avant/après, pas par un
  changement à l'aveugle.
- **Validation externe** : évaluer le modèle sur des images de scratch assay d'un autre
  laboratoire/microscope demanderait un nouveau jeu de données annoté, non disponible à ce stade —
  perspective la plus utile pour renforcer la généralisation, mais qui dépasse une simple
  itération de code.
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
python -m src.evaluate_ensemble --split test                         # évaluation ensemble de 4 modèles (voir Résultats)
python -m src.baseline --split test                                 # évaluation baseline
python -m src.statistical_comparison                                 # test de Wilcoxon U-Net vs baseline
python -m src.error_analysis_figures                                 # figures d'analyse d'erreur
python -m src.error_analysis_categories                              # analyse d'erreur quantitative par catégorie
python -m src.plot_training_curves --run_name unet_resnet34_holdout  # courbes d'apprentissage
python -m chatbot.evaluate_retrieval                                 # benchmark de récupération du RAG
```

Résultats bruts : `outputs/predictions/test_metrics.json`,
`outputs/predictions/test_metrics_tta.json`,
`outputs/predictions/test_ensemble_metrics.json`,
`outputs/predictions/test_baseline_metrics.json`,
`outputs/predictions/test_statistical_comparison.json`,
`outputs/predictions/test_error_by_category.csv`,
`outputs/predictions/test_error_category_summary.csv`,
`outputs/predictions/test_comparisons/`, `outputs/predictions/test_comparisons_tta/`,
`outputs/predictions/test_baseline_comparisons/`, `outputs/figures/`,
`chatbot/retrieval_eval_results.json`.
