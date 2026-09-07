"""Assistant IA : répond aux questions du biologiste en confrontant le résultat
de segmentation obtenu à la littérature scientifique sur la cicatrisation.

Utilise un LLM local via Ollama (modèle mistral) pour ne dépendre d'aucune clé
d'API externe, et index_articles.chercher() pour le contexte scientifique
(retrieval-augmented generation).

Usage:
    python chatbot/chatbot.py
"""
import queue
import sys
import threading
import time
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
    articles = chercher(question, n=n_articles)
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

Extraits d'articles scientifiques pertinents sur la cicatrisation :
{contexte}

Rédige un résumé scientifique structuré EN EXACTEMENT 4 sections, avec ces 4 titres en gras et \
dans cet ordre (rien avant, rien après, aucune section fusionnée ni omise) — chaque section ne \
doit contenir QUE le type d'information indiqué, pour que le biologiste distingue toujours ce qui \
est mesuré, ce qui est calculé, ce qui reste prudent, et ce qui vient de la littérature :

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
  N'écris jamais une vitesse qui ne soit pas recopiée telle quelle depuis ces deux sections.
- Reste un calcul, pas une interprétation biologique : ne parle pas encore ici d'accélération,
  de ralentissement, ni de ce que ça signifie pour la cicatrisation (voir section suivante).

**Interprétation prudente**
- Explique ce que ce calcul permet réellement de dire : les vitesses par intervalle ne comparent
  que des taux moyens entre intervalles, ça ne démontre ni ne mesure une accélération ou un
  ralentissement biologique de la cicatrisation (ça supposerait des mesures à l'intérieur de
  chaque intervalle, qu'on n'a pas). S'il y a une ligne "Intervalle le plus rapide / le plus lent"
  dans la section "Vitesse de fermeture par intervalle" ci-dessus, REPRENDS-LA TELLE QUELLE dans
  cet ordre (le plus rapide en premier) : n'inverse jamais cet ordre et ne recalcule/ne compare
  jamais toi-même quel intervalle est le plus rapide, cette ligne fait seule autorité. Utilise le
  patron "le taux moyen de fermeture estimé sur l'intervalle [bornes du plus rapide]
  ([vitesse du plus rapide]%/h) est supérieur à celui estimé sur l'intervalle [bornes du plus lent]
  ([vitesse du plus lent]%/h). Cette différence décrit uniquement les mesures disponibles et ne
  permet pas, à elle seule, de conclure à une accélération biologique de la cicatrisation." EN
  REPRENANT LES BORNES ET VITESSES EXACTES DE LA SECTION "Vitesse de fermeture par intervalle"
  CI-DESSUS, jamais des valeurs numériques différentes ; n'écris JAMAIS les mots "accélération",
  "accéléré", "ralentissement" ou "ralenti" au sujet de la cicatrisation elle-même, et n'écris
  jamais que "la plaie cicatrise plus vite/plus lentement",
- rappelle en une phrase, SANS RÉPÉTER LE CHIFFRE (déjà donné dans "Résultats observés" ci-dessus,
  ne le recompte ni ne l'invente pas une seconde fois ici), que ce résumé décrit une évolution
  globale sur peu de mesures, pas une cinétique fine,
- n'affirme jamais une tendance qui contredirait les vitesses par intervalle fournies ci-dessus.

**Mise en contexte scientifique**
- Si et seulement si un ou plusieurs extraits ci-dessus apportent un éclairage pertinent, cite
  leur titre complet entre guillemets (jamais "le premier article") et résume ce qu'ils apportent
  comme contexte général (mécanismes, méthodes) — jamais de lien ni d'URL (les extraits n'en
  fournissent pas). N'affirme JAMAIS que les mesures sont "conformes à la littérature", "en accord
  avec les études", qu'il s'agit d'un "résultat normal", ou que les extraits "confirment" tes
  chiffres : les extraits sont en général des résumés généraux, pas des points de comparaison
  chiffrés dans des conditions expérimentales comparables aux tiennes. Dis explicitement que les
  extraits apportent un contexte général mais ne permettent pas de conclure à une concordance
  quantitative avec les valeurs mesurées, sauf si un extrait décrit vraiment des conditions et une
  valeur chiffrée directement comparables (cas rare).
- IMPORTANT : si aucun extrait ci-dessus n'apporte d'éclairage réellement pertinent pour cette
  expérience précise, écris-le explicitement (ex : "le corpus documentaire disponible ne permet
  pas de mettre ces résultats en contexte de façon pertinente ici") plutôt que de forcer un lien
  approximatif avec un article qui ne correspond pas, et plutôt que de compléter avec des
  connaissances supposées non présentes dans les extraits ci-dessus.

Reste factuel et concis (5 à 9 phrases au total sur les 4 sections), sans réclamer d'information
supplémentaire au biologiste.

RÈGLE GLOBALE VALABLE DANS LES 4 SECTIONS CI-DESSUS, PAS SEULEMENT DANS "Interprétation prudente" \
: n'écris JAMAIS les mots "accélération", "accéléré", "ralentissement" ou "ralenti" au sujet de la \
cicatrisation ou de sa vitesse, y compris dans "Analyse quantitative" où seule une différence de \
taux moyens entre intervalles peut être mentionnée (jamais qu'un taux "accélère" ou "ralentit").
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

    lignes = [
        f"- Entre {_fr(iv['t_debut'])}h et {_fr(iv['t_fin'])}h : fermeture +{_fr(iv['delta_closure_pct'])} "
        f"points de %, soit une vitesse de {_fr(iv['vitesse_pct_h'])}%/h sur cet intervalle"
        for iv in intervalles
    ]
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


def generer_resume_stream(rows: list[dict], n_articles: int = 3):
    """Génère un résumé scientifique (style "Résultats") de l'évolution de la surface
    de la plaie au cours du temps, appuyé sur la littérature, en streaming.

    rows: sortie de kinetics.compute_kinetics_from_images (liste de dict avec au
    moins time_h, area_px2, closure_pct, triée par temps croissant).

    Retourne (générateur de morceaux de texte, sources).
    """
    requete_litterature = "wound healing scratch assay closure rate over time cell migration"
    articles = chercher(requete_litterature, n=n_articles)

    synthese = _calculer_synthese(rows)
    intervalles = _calculer_intervalles(rows)
    prompt = PROMPT_RAPPORT.format(
        synthese=_format_synthese(synthese, n_mesures=len(rows)),
        intervalles=_format_intervalles(intervalles),
        phrase_heterogeneite=_phrase_heterogeneite(synthese, intervalles),
        mesures=_format_mesures(rows),
        contexte=_format_contexte(articles),
    )
    return _stream_ollama(prompt), [a["titre"] for a in articles]


if __name__ == "__main__":
    resultat_test = {"surface_px2": 12450, "fermeture_pct": 42.0, "duree_h": 48}
    question_test = "Cette vitesse de fermeture est-elle cohérente avec la littérature ?"
    resultat = repondre(resultat_test, question_test)
    print(resultat["reponse"])
    print("\nSources :", ", ".join(resultat["sources"]))
