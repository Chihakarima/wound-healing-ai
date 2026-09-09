"""Assistant IA : répond aux questions du biologiste en confrontant le résultat
de segmentation obtenu à la littérature scientifique sur la cicatrisation.

Utilise un LLM local via Ollama (modèle mistral) pour ne dépendre d'aucune clé
d'API externe, et index_articles.chercher() pour le contexte scientifique
(retrieval-augmented generation).

Usage:
    python chatbot/chatbot.py
"""
import queue
import re
import sys
import threading
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).parent))
from index_articles import chercher

MODEL = "mistral"

# Délai max entre deux morceaux du flux avant d'abandonner (pas un délai sur la
# durée totale) : observé en pratique, Ollama peut se bloquer en cours de
# génération sans jamais renvoyer d'erreur ni de nouvel octet (probablement une
# contention GPU quand une segmentation d'image PyTorch vient de tourner juste
# avant dans la même session). Le timeout intégré au client HTTP d'ollama-python
# ne suffit pas à détecter ça (testé : blocage de 180s+ sans qu'il se déclenche
# pour un appel en streaming), donc la génération tourne dans un thread séparé
# et ce module surveille lui-même, via une file, qu'un nouveau morceau arrive
# bien avant CHUNK_TIMEOUT_S secondes.
CHUNK_TIMEOUT_S = 45.0
_client = ollama.Client()

PROMPT_TEMPLATE = """Tu es un assistant scientifique qui aide un biologiste à interpréter un \
résultat de segmentation automatique de plaie (wound healing assay), en répondant à sa question.

Résultat de segmentation obtenu par le biologiste :
{resultat_segmentation}

Question du biologiste :
{question}

Extraits d'articles scientifiques pertinents :
{contexte}

Réponds en français simple, clair, pour un biologiste non spécialiste en IA. Si le message est \
une vraie question scientifique, structure TOUJOURS ta réponse en EXACTEMENT 4 sections, avec ces \
4 titres en gras et dans cet ordre (rien avant, rien après, aucune section fusionnée ni omise) — \
chaque section ne doit contenir QUE le type d'information indiqué, pour que le biologiste \
distingue toujours ce qui est mesuré, ce qui en est déduit, ce qui reste incertain, et ce qui \
vient de la littérature :

**📊 Résultats observés**
- S'il y a un résultat de segmentation ci-dessus, reprends UNIQUEMENT les valeurs mesurées
  pertinentes pour la question, telles quelles (ne les recalcule pas, ne les arrondis pas
  différemment). Aucune interprétation ici.
- S'il n'y a pas de résultat ci-dessus (question générale, aucune image analysée), écris
  "Aucune image analysée pour cette question." et rien d'autre dans cette section.

**🔬 Interprétation**
- Réponds directement à la question du biologiste, en t'appuyant sur les résultats ci-dessus
  (si présents) et sur les extraits scientifiques (mécanismes, méthodes, ordres de grandeur
  généraux). Reste prudent : décris ce que les données/la littérature suggèrent, pas une
  certitude.

**⚠️ Limites**
- Dis explicitement ce que la question posée et les données disponibles ne permettent PAS de
  conclure (ex : nombre de mesures insuffisant pour une cinétique fine, absence de réplicats,
  extraits trop généraux pour une comparaison chiffrée directe). Une phrase suffit si les
  limites sont déjà couvertes ailleurs dans la réponse.

**📚 Littérature**
- Cite toujours le titre complet de l'article entre guillemets à chaque mention. Ne dis jamais
  "le premier article" ou "le deuxième article" : ces numéros ne correspondent à rien pour le
  lecteur et prêtent à confusion. N'invente jamais de lien ni d'URL (les extraits n'en
  fournissent pas).
- N'affirme JAMAIS qu'un résultat est "conforme à la littérature", "en accord avec les études",
  un "résultat normal", ou qu'il "confirme" quoi que ce soit scientifiquement, sauf si un extrait
  décrit explicitement des conditions expérimentales comparables (même type de test, échelle de
  temps comparable) ET une valeur chiffrée directement comparable. C'est le cas le plus rare : les
  extraits sont en général des résumés généraux (mécanismes, méthodes), pas des points de
  comparaison chiffrés pour les conditions précises du biologiste. Par défaut, dis explicitement
  que les extraits fournissent un contexte général mais ne permettent pas de conclure à une
  concordance quantitative, plutôt que d'inventer un rapprochement.
- Si aucun extrait ci-dessus n'apporte d'éclairage réellement pertinent pour cette question
  précise, écris-le explicitement (ex : "le corpus documentaire disponible ne permet pas de
  mettre ce point en contexte de façon pertinente ici") plutôt que de forcer un lien approximatif.

Consignes valables dans les 4 sections :
- Ne pose jamais de question de clarification en retour (pas de "pouvez-vous préciser...",
  "quelle est la durée de l'expérience ?", etc.) : réponds directement avec le résultat de
  segmentation et les extraits déjà fournis ci-dessus. S'il manque une information pour répondre
  complètement, dis-le en une phrase (section Limites) et réponds quand même du mieux possible
  avec ce qui est disponible, au lieu de renvoyer la question au biologiste.
- Reste factuel et concis (une à trois phrases par section).

Exception à la structure en 4 sections : si le message n'est pas une vraie question scientifique
(simple salutation comme "bonjour", message vide de sens, remerciement...), ignore tout ce qui
précède et réponds brièvement et simplement (ex : salue en retour, invite à poser une question
sur la cicatrisation), sans sections ni lien forcé avec les extraits.
"""


def _format_resultat(resultat_segmentation: dict | None) -> str:
    if not resultat_segmentation:
        return "(aucun : question générale, sans image analysée)"
    return "\n".join(f"- {cle} : {valeur}" for cle, valeur in resultat_segmentation.items())


def _format_contexte(articles: list[dict]) -> str:
    # Pas de crochets autour du titre : un modèle comme mistral a tendance à
    # recopier ce gabarit tel quel comme "citation", au lieu de suivre la
    # consigne de citer le titre entre guillemets dans sa réponse.
    return "\n\n".join(
        f"Titre : {a['titre']}\nAnnée : {a['annee']}\nExtrait : {a['extrait']}"
        for a in articles
    )


_ARRET = object()  # sentinelle : signale la fin normale du flux dans la file


BATCH_INTERVAL_S = 0.5  # regroupe les morceaux reçus sur cette fenêtre avant de les afficher


def _stream_ollama(prompt: str):
    """Lance une génération Ollama en streaming ; lève RuntimeError immédiatement
    si Ollama est injoignable (avant de renvoyer le générateur de morceaux).

    La génération tourne dans un thread à part qui pousse chaque morceau dans une
    file ; ce générateur lit la file avec un timeout et regroupe les morceaux reçus
    sur BATCH_INTERVAL_S avant de les céder. Le regroupement est nécessaire : testé
    en conditions réelles, céder un morceau à chaque token (observé : ~260 morceaux
    sur ~60s) fait décrocher l'affichage st.write_stream côté navigateur en cours de
    route (Streamlit cesse de transmettre les mises à jour) alors que la génération
    elle-même se termine normalement côté serveur — un thread de debug l'a confirmé.

    Si Ollama se bloque en cours de route (aucun nouveau morceau pendant plus de
    CHUNK_TIMEOUT_S), on abandonne proprement avec un message d'avertissement ajouté
    au texte déjà reçu, plutôt que de laisser l'appli figée indéfiniment. Le thread
    bloqué est laissé mourir en arrière-plan (démon) : on n'attend pas qu'il se
    débloque."""
    try:
        stream = _client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Vérifiez qu'Ollama est installé et lancé "
            f"(ollama serve) et que le modèle est disponible (ollama pull {MODEL})."
        ) from exc

    file_morceaux = queue.Queue()

    def producteur():
        try:
            for chunk in stream:
                file_morceaux.put(chunk["message"]["content"])
        except Exception:
            pass
        finally:
            file_morceaux.put(_ARRET)

    threading.Thread(target=producteur, daemon=True).start()

    def morceaux():
        # Sondage court (POLL_S) pour pouvoir vérifier régulièrement à la fois le
        # délai depuis le dernier morceau reçu (détection de blocage, CHUNK_TIMEOUT_S)
        # et le délai depuis le dernier envoi au navigateur (regroupement d'affichage,
        # BATCH_INTERVAL_S) sans dépendre d'un seul timeout pour les deux.
        POLL_S = 0.1
        tampon = []
        dernier_item = time.monotonic()
        dernier_envoi = time.monotonic()

        while True:
            try:
                item = file_morceaux.get(timeout=POLL_S)
            except queue.Empty:
                item = None

            maintenant = time.monotonic()

            if item is not None:
                if item is _ARRET:
                    if tampon:
                        yield "".join(tampon)
                    return
                tampon.append(item)
                dernier_item = maintenant

            if maintenant - dernier_item > CHUNK_TIMEOUT_S:
                if tampon:
                    yield "".join(tampon)
                yield (
                    "\n\n⚠️ *Génération interrompue : le modèle local n'a pas répondu à temps "
                    "(Ollama a peut-être été surchargé, par exemple juste après une analyse "
                    "d'image). Réessayez.*"
                )
                return

            if tampon and maintenant - dernier_envoi >= BATCH_INTERVAL_S:
                yield "".join(tampon)
                tampon = []
                dernier_envoi = maintenant

    return morceaux()


# Repère les citations entre guillemets (français « » ou droits/courbes "..."/"...")
# assez longues pour être un titre d'article plutôt qu'un mot mis en emphase.
_CITATION_RE = re.compile(r"[«\"“]([^»\"”]{15,300})[»\"”]")
SEUIL_SIMILARITE_CITATION = 0.7  # ratio SequenceMatcher au-dessus duquel deux titres sont "le même"


def _normaliser_titre(texte: str) -> str:
    """Normalise un titre pour la comparaison : enlève les accents/diacritiques
    (variantes unicode entre le titre stocké et sa recopie par le LLM), casse et
    espaces superflus."""
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return " ".join(texte.lower().split())


def detecter_citations_suspectes(texte: str, sources: list[str]) -> list[str]:
    """Repère, dans un texte déjà généré, les citations entre guillemets qui ne
    correspondent à aucun des titres réellement retournés par le retrieval
    (`sources`) -- signe qu'un titre a été inventé par le LLM plutôt que recopié
    depuis le contexte fourni (observé en usage réel : un titre plausible mais
    absent du corpus, voir chatbot_prompt_fragility en mémoire projet).

    Vérification purement programmatique APRÈS génération : ne modifie ni le
    prompt ni la génération elle-même, donc n'ajoute aucun des risques déjà
    documentés (toute modification du contexte/prompt doit être validée par un
    test d'isolation ; une simple relecture du texte produit n'a pas cette
    contrainte).

    Retourne la liste des citations suspectes trouvées dans `texte` (vide si
    aucune, y compris si `texte` ne contient aucune citation entre guillemets).
    """
    sources_norm = [_normaliser_titre(s) for s in sources]
    suspectes = []
    for citation in _CITATION_RE.findall(texte):
        citation_norm = _normaliser_titre(citation)
        correspond_a_une_source = any(
            citation_norm in source_norm
            or source_norm in citation_norm
            or SequenceMatcher(None, citation_norm, source_norm).ratio() >= SEUIL_SIMILARITE_CITATION
            for source_norm in sources_norm
        )
        if not correspond_a_une_source:
            suspectes.append(citation.strip())
    return suspectes


LONGUEUR_MIN_QUESTION = 8  # en dessous, ce n'est pas une vraie question (salutation, faute de frappe...)

REPONSE_SALUTATION = (
    "Bonjour ! Posez-moi une question sur la cicatrisation ou sur un résultat de "
    "segmentation (surface, vitesse de fermeture...) et je chercherai la littérature "
    "scientifique pertinente pour vous répondre."
)


def _question_triviale(question: str) -> bool:
    """Un message trop court pour être une vraie question (ex: "bnj", "ok") ne doit
    pas déclencher toute la recherche RAG + génération : un petit modèle local comme
    mistral 7B ne suit pas de façon fiable la consigne "ne force pas de réponse
    scientifique à une salutation", donc ce filtre est fait en code plutôt qu'en
    comptant sur le LLM."""
    return len(question.strip()) < LONGUEUR_MIN_QUESTION


def _construire_prompt(resultat_segmentation: dict | None, question: str, n_articles: int):
    # hybride=False (décision finale, 2026-09-08) : activé un temps sur demande
    # explicite malgré un risque mesuré (voir chatbot_prompt_fragility en mémoire
    # projet), puis revenu en arrière après reconsidération -- pour un usage
    # scientifique sérieux, le risque de chiffres fabriqués l'emporte sur le gain
    # de retrieval. Le mode hybride reste disponible et validé pour le retrieval
    # seul (voir evaluate_retrieval.py), pas pour la génération.
    articles = chercher(question, n=n_articles, hybride=False)
    prompt = PROMPT_TEMPLATE.format(
        resultat_segmentation=_format_resultat(resultat_segmentation),
        question=question,
        contexte=_format_contexte(articles),
    )
    return prompt, articles


def repondre(resultat_segmentation: dict | None, question: str, n_articles: int = 3) -> dict:
    """Génère une réponse à la question du biologiste, appuyée sur la littérature.

    resultat_segmentation: dict des métriques déjà calculées par l'app
    (ex: {"surface_px2": 12450, "fermeture_pct": 42.0, "duree_h": 48}), ou None
    pour une question générale sans image analysée.

    Retourne {"reponse": str, "sources": [titres d'articles utilisés]}.
    """
    if _question_triviale(question):
        return {"reponse": REPONSE_SALUTATION, "sources": []}

    prompt, articles = _construire_prompt(resultat_segmentation, question, n_articles)

    try:
        response = _client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    except Exception as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Vérifiez qu'Ollama est installé et lancé "
            f"(ollama serve) et que le modèle est disponible (ollama pull {MODEL})."
        ) from exc

    return {
        "reponse": response["message"]["content"],
        "sources": [a["titre"] for a in articles],
    }


def repondre_stream(resultat_segmentation: dict | None, question: str, n_articles: int = 3):
    """Comme repondre(), mais renvoie la réponse morceau par morceau au lieu
    d'attendre le texte complet : une réponse mistral en local prend 60-90s,
    et un écran figé pendant tout ce temps donne l'impression que l'appli est
    plantée. Affichée en direct (ex: st.write_stream côté app.py), la réponse
    apparaît mot à mot dès les premières secondes, comme dans ChatGPT.

    Retourne (générateur de morceaux de texte, sources). Lève RuntimeError
    immédiatement si Ollama est injoignable (avant de renvoyer le générateur).
    """
    if _question_triviale(question):
        return iter([REPONSE_SALUTATION]), []

    prompt, articles = _construire_prompt(resultat_segmentation, question, n_articles)
    return _stream_ollama(prompt), [a["titre"] for a in articles]


PROMPT_RAPPORT = """Tu es un assistant scientifique qui aide un biologiste à rédiger un premier \
résumé de son expérience de cicatrisation de plaie (wound healing / scratch assay), à partir de \
mesures de surface obtenues automatiquement par segmentation d'image sur une série d'images prises \
à différents temps.

Synthèse chiffrée déjà calculée par le pipeline (valeurs finales à reprendre telles quelles dans \
ta réponse ; ne recalcule et n'invente aucun de ces chiffres, ne les arrondis pas différemment) :
{synthese}

Vitesse de fermeture par intervalle, déjà calculée par le pipeline (sers-t'en pour comparer les \
vitesses moyennes entre intervalles, sans recalculer ces vitesses toi-même) :
{intervalles}

Phrase de liaison entre la vitesse moyenne globale et l'hétérogénéité par intervalle, déjà \
composée par le pipeline (à reprendre mot pour mot dans "Analyse quantitative" si elle contient \
une vraie phrase ; si elle est entre parenthèses, ne la reprends pas et n'en invente pas \
d'équivalent) :
{phrase_heterogeneite}

Détail des mesures par point de temps (pour décrire la tendance point par point uniquement) :
{mesures}

Phrase de comparaison des intervalles, déjà composée par le pipeline (à reprendre mot pour mot \
comme première phrase de "Interprétation prudente" si elle contient une vraie phrase ; si elle est \
entre parenthèses, n'invente pas de comparaison à la place) :
{phrase_interpretation}

Réouverture partielle éventuelle, déjà détectée par le pipeline (à reprendre mot pour mot dans \
"Interprétation prudente" si elle contient une vraie phrase ; si elle est entre parenthèses, \
n'invente pas de réouverture à la place) :
{phrase_reouverture}

Rédige un résumé scientifique structuré EN EXACTEMENT 3 sections, avec ces 3 titres en gras et \
dans cet ordre (rien avant, rien après, aucune section fusionnée ni omise) — chaque section ne \
doit contenir QUE le type d'information indiqué, pour que le biologiste distingue toujours ce qui \
est mesuré, ce qui est calculé, et ce qui reste prudent :

**Résultats observés**
- 1 à 2 phrases, UNIQUEMENT les valeurs mesurées telles quelles : nombre de mesures, surface
  initiale et finale, fermeture finale en %, durée totale d'observation. Reprends-les exactement
  depuis la synthèse chiffrée ci-dessus (ligne "Nombre de mesures" incluse), sans les recalculer
  ni en changer aucune. Aucune interprétation ici.
- n'affirme jamais que "le biologiste a effectué l'expérience" ou toute variante supposant qui a
  réalisé la manipulation (l'application ne le sait pas) : décris les mesures elles-mêmes, par
  exemple "les mesures analysées proviennent d'un essai de type scratch assay" ou "l'analyse des
  images issues d'un scratch assay montre une diminution de la surface non colonisée".

**Analyse quantitative**
- Cite TOUJOURS la vitesse moyenne de fermeture sur l'ensemble de la période (depuis la synthèse
  chiffrée), même si tu détailles ensuite les taux par intervalle : ne la laisse jamais implicite
  ou absente. Précise aussi quel intervalle a le taux le plus élevé / le plus faible (depuis les
  vitesses par intervalle ci-dessus), sans te contenter de répéter les chiffres déjà visibles dans
  le tableau du biologiste.
- Si la "Phrase de liaison" ci-dessus contient une vraie phrase (pas une note entre parenthèses),
  REPRENDS-LA MOT POUR MOT dans cette section, sans changer un seul chiffre et sans la recomposer
  toi-même : ne recopie jamais le taux d'un intervalle à la place de la vitesse moyenne globale dans
  cette phrase, cette phrase fait seule autorité sur ce point précis. Sans elle, un lecteur qui compare le
  résumé au tableau des mesures peut croire à un oubli plutôt qu'à un choix de présentation. Si elle
  est entre parenthèses, n'invente pas de comparaison d'hétérogénéité à la place.
- 2 à 3 phrases au total pour cette section.
- IMPORTANT : chaque vitesse en %/h que tu écris doit être copiée mot pour mot depuis la section
  "Vitesse de fermeture par intervalle" ci-dessus (ou depuis la synthèse chiffrée pour la vitesse
  moyenne) : ce sont les deux seuls endroits de ce prompt qui font autorité sur les vitesses.
  N'écris jamais une vitesse qui ne soit pas recopiée telle quelle depuis ces deux sections, SIGNE
  INCLUS : recopie le signe "-" si la source en affiche un (une vitesse négative signifie que la
  surface a augmenté, pas seulement fermé plus lentement), et n'en ajoute JAMAIS un si la source
  n'en affiche pas -- dans les deux cas, le signe de ta phrase doit être identique à celui affiché
  dans la source, jamais déduit ni copié depuis un autre exemple.
- Reste un calcul, pas une interprétation biologique : ne parle pas encore ici d'accélération,
  de ralentissement, ni de ce que ça signifie pour la cicatrisation (voir section suivante).

**Interprétation prudente**
- S'il y a une "Phrase de comparaison des intervalles" fournie plus bas (pas une note entre
  parenthèses), REPRENDS-LA MOT POUR MOT comme première phrase de cette section, SANS CHANGER NI
  RECALCULER NI RECOMPOSER un seul mot, un seul chiffre, un seul signe ou l'ordre des bornes : cette
  phrase fait seule autorité, ne la remplace jamais par ta propre comparaison des intervalles même
  si tu penses pouvoir la reformuler plus clairement. Si elle est entre parenthèses, n'invente aucune
  comparaison à la place. N'écris JAMAIS les mots "accélération", "accéléré", "ralentissement" ou
  "ralenti" au sujet de la cicatrisation elle-même ailleurs dans cette section, et n'écris jamais que
  "la plaie cicatrise plus vite/plus lentement",
- Si la "Réouverture partielle" ci-dessus contient une vraie phrase (pas une note entre
  parenthèses), REPRENDS-LA MOT POUR MOT, EN PLUS du reste de cette section, sans changer un seul
  chiffre et sans la recomposer toi-même : c'est une information factuelle importante pour le
  biologiste (la surface non colonisée a augmenté, pas seulement "ralenti"), jamais à passer sous
  silence ni à reformuler en "ralentissement". Si elle est entre parenthèses, n'invente aucune
  réouverture à la place.
- rappelle en une phrase, SANS RÉPÉTER LE CHIFFRE (déjà donné dans "Résultats observés" ci-dessus,
  ne le recompte ni ne l'invente pas une seconde fois ici), que ce résumé décrit une évolution
  globale sur peu de mesures, pas une cinétique fine,
- n'affirme jamais une tendance qui contredirait les vitesses par intervalle fournies ci-dessus.

Reste factuel et concis (4 à 7 phrases au total sur les 3 sections), sans réclamer d'information
supplémentaire au biologiste.

RÈGLE GLOBALE VALABLE DANS LES 3 SECTIONS CI-DESSUS, PAS SEULEMENT DANS "Interprétation prudente" \
: n'écris JAMAIS les mots "accélération", "accéléré", "ralentissement" ou "ralenti" au sujet de la \
cicatrisation ou de sa vitesse, y compris dans "Analyse quantitative" où seule une différence de \
taux moyens entre intervalles peut être mentionnée (jamais qu'un taux "accélère" ou "ralentit").
"""


# Appel isolé, séparé de PROMPT_RAPPORT : cette section a provoqué à deux reprises des
# fabrications de chiffres dans les sections numériques quand elle partageait le même appel
# LLM que les mesures (patron ouvert à trous, puis renforcement en prose -- voir
# chatbot_prompt_fragility en mémoire projet et "Fiabilité de la génération par LLM" dans le
# README). PROMPT_CONTEXTE ne reçoit jamais aucune mesure : la contamination croisée devient
# structurellement impossible, pas seulement évitée par une consigne de prompt.
PROMPT_CONTEXTE = """Tu es un assistant scientifique qui rédige la mise en contexte \
bibliographique d'un résumé d'expérience de cicatrisation de plaie (wound healing / scratch \
assay), à partir d'extraits d'articles déjà sélectionnés pour leur pertinence.

Extraits d'articles scientifiques pertinents :
{contexte}

Rédige un paragraphe COURT, 3 phrases MAXIMUM au total (jamais plus, même si plusieurs extraits \
sont pertinents) — PAS une phrase ni un paragraphe par article, ce paragraphe reste une synthèse \
groupée, jamais une revue de littérature détaillée :
- Si un ou plusieurs extraits ci-dessus apportent un éclairage pertinent, UNE SEULE phrase cite
  ensemble leur(s) titre(s) complet(s) entre guillemets (jamais "le premier article") et résume en
  bloc ce qu'ils établissent concrètement (méthode d'imagerie quantitative, mécanisme de migration
  cellulaire, intérêt d'un suivi temporel de la fermeture...) — dis CE QU'ils établissent, pas
  seulement QU'ils existent, et ne détaille jamais leur méthodologie article par article. Jamais de
  lien ni d'URL (les extraits n'en fournissent pas).
- N'affirme JAMAIS qu'un résultat est "conforme à la littérature", "en accord avec les études", un
  "résultat normal", ou que les extraits "confirment" quoi que ce soit : les extraits sont en
  général des résumés généraux, pas des points de comparaison chiffrés dans des conditions
  comparables.
- Termine TOUJOURS par une phrase disant explicitement que ces travaux ne fournissent pas de
  valeur de référence directement comparable à cette expérience précise (protocole, système
  d'imagerie et conditions biologiques différents), et que toute mesure obtenue par ailleurs décrit
  sa propre série expérimentale, pas une comparaison à une cinétique biologique universelle.
- Si aucun extrait ci-dessus n'apporte d'éclairage réellement pertinent, écris-le explicitement en
  une phrase (ex : "le corpus documentaire disponible ne permet pas de mettre ces résultats en
  contexte de façon pertinente ici") plutôt que de forcer un lien approximatif avec un article qui
  ne correspond pas — dans ce cas, le paragraphe entier ne fait qu'une phrase.
- Tu n'as accès à aucune mesure de surface, de fermeture ni de vitesse : n'en mentionne ni n'en
  invente aucune, ce paragraphe ne parle que de ce que montrent les extraits ci-dessus.
"""


def _fr(nombre: float) -> str:
    """Formate un nombre avec une virgule décimale (convention française), plutôt
    que le point par défaut de Python (ex: 0.0h -> 0h, 1.58 -> 1,58) : ce texte
    est injecté tel quel dans les réponses du chatbot, en français."""
    if float(nombre) == int(nombre):
        return str(int(nombre))
    return str(nombre).replace(".", ",")


def _format_mesures(rows: list[dict]) -> str:
    lignes = []
    for r in rows:
        ligne = f"- T = {_fr(r['time_h'])}h : {r['area_px2']} px²"
        if "area_cm2" in r:
            ligne += f" ({_fr(r['area_cm2'])} cm²)"
        ligne += f", fermeture {_fr(r['closure_pct'])}% par rapport à T0"
        lignes.append(ligne)
    return "\n".join(lignes)


def _calculer_synthese(rows: list[dict]) -> dict:
    """Calcule les chiffres clés (surface initiale/finale, fermeture finale, durée,
    vitesse moyenne) directement dans le code plutôt que de laisser le LLM les
    déduire : un modèle local comme mistral 7B fait régulièrement des erreurs
    arithmétiques sur ce type de calcul (constaté : fermeture et vitesse moyenne
    erronées par rapport aux mesures fournies)."""
    t0, t_final = rows[0], rows[-1]
    duree_h = t_final["time_h"] - t0["time_h"]
    fermeture_finale_pct = t_final["closure_pct"]
    vitesse_moyenne_pct_h = round(fermeture_finale_pct / duree_h, 2) if duree_h > 0 else 0.0
    return {
        "t0": t0,
        "t_final": t_final,
        "duree_h": duree_h,
        "fermeture_finale_pct": fermeture_finale_pct,
        "vitesse_moyenne_pct_h": vitesse_moyenne_pct_h,
    }


def _format_synthese(synthese: dict, n_mesures: int) -> str:
    t0, t_final = synthese["t0"], synthese["t_final"]

    def _surface(r):
        s = f"{r['area_px2']} px²"
        if "area_cm2" in r:
            s += f" ({_fr(r['area_cm2'])} cm²)"
        return s

    return "\n".join([
        f"- Nombre de mesures : {n_mesures} (décrit une évolution globale, pas une cinétique fine)",
        f"- Surface initiale (T = {_fr(t0['time_h'])}h) : {_surface(t0)}",
        f"- Surface finale (T = {_fr(t_final['time_h'])}h) : {_surface(t_final)}",
        f"- Fermeture finale : {_fr(synthese['fermeture_finale_pct'])}% par rapport à T0",
        f"- Durée totale d'observation : {_fr(synthese['duree_h'])}h",
        f"- Vitesse moyenne de fermeture sur la période observée : {_fr(synthese['vitesse_moyenne_pct_h'])}%/h "
        "(points de % par heure, pas une vitesse instantanée)",
    ])


def _calculer_intervalles(rows: list[dict]) -> list[dict]:
    """Calcule la vitesse de fermeture entre chaque paire de points consécutifs.

    Sans ça, dire "la fermeture a ralenti entre 24h et 48h" oblige le LLM à comparer
    les points lui-même à partir des seules valeurs cumulées par rapport à T0, avec
    le même risque d'erreur d'interprétation que pour les chiffres globaux (constaté
    en pratique). En calculant la vitesse de chaque intervalle dans le code, le LLM
    n'a plus qu'à la commenter, pas à la déduire."""
    intervalles = []
    for prev, curr in zip(rows, rows[1:]):
        duree_h = curr["time_h"] - prev["time_h"]
        delta_closure_pct = round(curr["closure_pct"] - prev["closure_pct"], 1)
        vitesse_pct_h = round(delta_closure_pct / duree_h, 2) if duree_h > 0 else 0.0
        intervalles.append({
            "t_debut": prev["time_h"],
            "t_fin": curr["time_h"],
            "duree_h": duree_h,
            "delta_closure_pct": delta_closure_pct,
            "vitesse_pct_h": vitesse_pct_h,
        })
    return intervalles


def _format_intervalles(intervalles: list[dict]) -> str:
    if not intervalles:
        return "(un seul point de mesure disponible : pas d'intervalle à comparer)"

    lignes = []
    for iv in intervalles:
        if iv["delta_closure_pct"] < 0:
            # Delta négatif : la surface non colonisée a augmenté sur cet intervalle
            # (la plaie s'est partiellement rouverte), pas juste "fermé plus lentement" --
            # signalé explicitement ici plutôt que de laisser un "+" trompeur devant un
            # nombre négatif (ex: "+-19,9 points de %", corrigé le 2026-09-09).
            lignes.append(
                f"- Entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h : réouverture partielle, "
                f"la surface non colonisée a augmenté de {_fr(abs(iv['delta_closure_pct']))} points de %, "
                f"soit une vitesse de {_fr(iv['vitesse_pct_h'])}%/h sur cet intervalle"
            )
        else:
            lignes.append(
                f"- Entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h : fermeture +{_fr(iv['delta_closure_pct'])} "
                f"points de %, soit une vitesse de {_fr(iv['vitesse_pct_h'])}%/h sur cet intervalle"
            )
    if len(intervalles) > 1:
        plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
        if plus_rapide is not plus_lent:
            lignes.append(
                f"- Intervalle le plus rapide : {_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h "
                f"({_fr(plus_rapide['vitesse_pct_h'])}%/h) ; le plus lent : "
                f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h)"
            )
    return "\n".join(lignes)


def _phrase_heterogeneite(synthese: dict, intervalles: list[dict]) -> str:
    """Phrase toute faite reliant le taux moyen global à l'hétérogénéité entre
    intervalles, composée dans le code plutôt que par le LLM : constaté en pratique,
    même avec les deux chiffres déjà disponibles séparément (taux moyen dans la
    synthèse, taux par intervalle dans _format_intervalles), un modèle local comme
    mistral 7B peut recopier le mauvais chiffre en recomposant cette phrase
    lui-même (ex: réutiliser le taux d'un intervalle à la place du taux global)."""
    if len(intervalles) < 2:
        return "(un seul intervalle disponible : pas de phrase de liaison pertinente)"

    plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    if plus_rapide is plus_lent:
        return "(taux identique sur tous les intervalles : pas d'hétérogénéité à signaler)"

    return (
        f"La vitesse moyenne de fermeture sur l'ensemble de la période "
        f"({_fr(synthese['vitesse_moyenne_pct_h'])}%/h) masque cette hétérogénéité : le taux "
        f"estimé est plus élevé sur l'intervalle {_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h "
        f"({_fr(plus_rapide['vitesse_pct_h'])}%/h) que sur l'intervalle "
        f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h)."
    )


def _phrase_interpretation(intervalles: list[dict]) -> str:
    """Phrase toute faite pour "Interprétation prudente", composée dans le code
    plutôt que laissée au LLM remplir un patron à trous. Constaté en pratique
    (2026-09-09) : même avec un patron explicite ("le taux ... est supérieur à
    celui ...") et une consigne de recopie stricte, mistral 7B fabrique parfois un
    signe "-" sur une vitesse pourtant positive, ou inverse l'ordre des bornes d'un
    intervalle -- un patron à trous laisse encore trop de place à l'erreur pour ce
    genre de comparaison numérique, contrairement à une phrase entièrement fournie
    (voir _phrase_heterogeneite, qui n'a jamais montré ce problème)."""
    if len(intervalles) < 2:
        return "(un seul intervalle disponible : pas de comparaison pertinente)"

    plus_rapide = max(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    plus_lent = min(intervalles, key=lambda iv: iv["vitesse_pct_h"])
    if plus_rapide is plus_lent:
        return "(taux identique sur tous les intervalles : pas de comparaison pertinente)"

    return (
        f"Le taux moyen de fermeture estimé sur l'intervalle "
        f"{_fr(plus_rapide['t_debut'])}h-{_fr(plus_rapide['t_fin'])}h ({_fr(plus_rapide['vitesse_pct_h'])}%/h) "
        f"est supérieur à celui estimé sur l'intervalle "
        f"{_fr(plus_lent['t_debut'])}h-{_fr(plus_lent['t_fin'])}h ({_fr(plus_lent['vitesse_pct_h'])}%/h). "
        "Cette différence décrit uniquement les mesures disponibles et ne permet pas, à elle seule, "
        "de conclure à une accélération biologique de la cicatrisation."
    )


def _phrase_reouverture(intervalles: list[dict]) -> str:
    """Phrase toute faite signalant qu'un intervalle a un delta de fermeture négatif
    (la surface non colonisée a augmenté : la plaie s'est partiellement rouverte),
    composée dans le code plutôt que laissée au LLM. Constaté en pratique (retour du
    biologiste, 2026-09-09) : le patron d'"Interprétation prudente" de PROMPT_RAPPORT
    compare seulement "intervalle le plus rapide" vs "le plus lent", ce qui, pour un
    delta négatif, se traduit en une simple différence de vitesse -- masquant le fait
    plus important qu'il ne s'agit pas d'un ralentissement mais d'une réouverture
    partielle de la plaie, une information que le biologiste doit voir explicitement
    (mesure suspecte à vérifier, ou vrai phénomène biologique à creuser)."""
    reouvertures = [iv for iv in intervalles if iv["delta_closure_pct"] < 0]
    if not reouvertures:
        return "(aucune réouverture observée sur les intervalles disponibles)"

    lignes = [
        f"la surface non colonisée a augmenté de {_fr(abs(iv['delta_closure_pct']))} points de % "
        f"entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h"
        for iv in reouvertures
    ]
    return (
        "Réouverture partielle détectée sur les mesures disponibles : " + " ; ".join(lignes) + ". "
        "Cela peut correspondre à un vrai phénomène biologique (ex: décollement cellulaire) ou à "
        "une variation de mesure (segmentation, éclairage) : ce résumé ne permet pas de trancher, "
        "une vérification de l'image concernée est recommandée."
    )


def generer_resume_stream(rows: list[dict], n_articles: int = 3):
    """Génère un résumé scientifique (style "Résultats") de l'évolution de la surface
    de la plaie au cours du temps, appuyé sur la littérature, en streaming.

    rows: sortie de kinetics.compute_kinetics_from_images (liste de dict avec au
    moins time_h, area_px2, closure_pct, triée par temps croissant).

    Retourne (générateur de morceaux de texte, sources).
    """
    # Reformulée le 2026-09-08 (voir chatbot_prompt_fragility, mémoire projet) : la
    # requête d'origine ("wound healing scratch assay closure rate over time cell
    # migration") ne retrouvait aucun article de la catégorie analyse_temporelle
    # dans le top 10, forçant systématiquement la section "Mise en contexte
    # scientifique" à répondre qu'elle ne peut rien dire -- honnête, mais peu
    # informatif. Ce vocabulaire, repris des résumés d'articles ciblés, retrouve
    # "Study of Wound Healing Dynamics by Single Pseudo-Particle Tracking..." au
    # rang 2. Isolation-testée 3/3 propre, gardée.
    #
    # Une 2e itération a aussi été essayée (vocabulaire repris de "The Frequent
    # Sampling of Wound Scratch Assay...", qui plaçait CE second article et le
    # précédent aux rangs 1 et 2 -- meilleur retrieval sur le papier) mais 3/3
    # générations ont alors fabriqué la vitesse moyenne et les vitesses par
    # intervalle (ex: 7,2%/h puis 0%/h au lieu de 0,83%/h et 1,58%/h) : cet article
    # contient lui-même beaucoup de chiffres de vitesse/cinétique dans son propre
    # résumé (fenêtre de 6h, doses de drogues, % de diminution de vitesse), ce qui
    # semble amener le LLM à les confondre avec les données de l'utilisateur.
    # Non gardée -- un meilleur retrieval n'implique pas une génération plus fiable.
    requete_litterature = (
        "wound healing assay reproducibility standardisation kinetics time-lapse "
        "phase contrast tracking cell migration dynamics"
    )
    # hybride=False (décision finale, 2026-09-08) : un test d'isolation avait
    # montré 3/3 générations en hybride fabriquant des bornes d'intervalle et un
    # taux de fermeture inexistants (ex: "36-48h", alors que ROWS n'a que
    # 0/24/48h) contre 2/3 correctes en embeddings seuls -- activé un temps sur
    # demande explicite malgré ce risque, puis revenu en arrière après
    # reconsidération (voir chatbot_prompt_fragility, mémoire projet) : pour un
    # usage scientifique sérieux, le risque de chiffres fabriqués l'emporte sur
    # le gain de retrieval. Mode hybride gardé uniquement comme option validée
    # pour les métriques de retrieval (voir evaluate_retrieval.py).
    articles = chercher(requete_litterature, n=n_articles, hybride=False)

    synthese = _calculer_synthese(rows)
    intervalles = _calculer_intervalles(rows)
    prompt_resultats = PROMPT_RAPPORT.format(
        synthese=_format_synthese(synthese, n_mesures=len(rows)),
        intervalles=_format_intervalles(intervalles),
        phrase_heterogeneite=_phrase_heterogeneite(synthese, intervalles),
        phrase_interpretation=_phrase_interpretation(intervalles),
        phrase_reouverture=_phrase_reouverture(intervalles),
        mesures=_format_mesures(rows),
    )
    prompt_contexte = PROMPT_CONTEXTE.format(contexte=_format_contexte(articles))

    # Lancé eagerly (pas dans le générateur ci-dessous) pour garder le contrat de
    # repondre_stream : lever RuntimeError immédiatement si Ollama est injoignable,
    # avant de renvoyer quoi que ce soit à l'appelant.
    flux_resultats = _stream_ollama(prompt_resultats)

    def flux():
        yield from flux_resultats
        yield "\n\n**Mise en contexte scientifique**\n"
        # Appel séparé, lancé seulement une fois le premier flux épuisé : ce prompt
        # ne contient aucune mesure (voir PROMPT_CONTEXTE), donc rien de ce résumé
        # numérique ne peut être contaminé par la génération de cette section.
        yield from _stream_ollama(prompt_contexte)

    return flux(), [a["titre"] for a in articles]


if __name__ == "__main__":
    resultat_test = {"surface_px2": 12450, "fermeture_pct": 42.0, "duree_h": 48}
    question_test = "Cette vitesse de fermeture est-elle cohérente avec la littérature ?"
    resultat = repondre(resultat_test, question_test)
    print(resultat["reponse"])
    print("\nSources :", ", ".join(resultat["sources"]))
