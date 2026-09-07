"""Interface Streamlit pour tester le modèle de segmentation de plaie.

Usage:
    streamlit run app.py
"""
import io
import os
import sys
import zipfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# streamlit-drawable-canvas 0.9.3 (dernière version publiée) appelle
# streamlit.elements.image.image_to_url(image, width, ...), une fonction que
# Streamlit récent a déplacée vers elements.lib.image_utils avec un paramètre
# layout_config au lieu d'un simple width. On restaure l'ancienne signature
# pour que le composant fonctionne avec la version de Streamlit installée.
import streamlit.elements.image as _st_image_module
if not hasattr(_st_image_module, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_v2
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _image_to_url_v2(image, _LayoutConfig(width=width), clamp, channels, output_format, image_id)

    _st_image_module.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatbot"))
from kinetics import build_figure, compute_kinetics_from_images, guess_time_h, save_csv, save_plot
from measure import area_px_to_cm2, extract_contours, wound_area_px
from metrics import confusion_counts, dice_score, iou_score
from predict import draw_contour_overlay, load_model, predict_mask
from preprocess import clean_image
from chatbot import generer_resume_stream, repondre_stream

st.set_page_config(page_title="Segmentation de plaie", page_icon="🩹", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --accent: #0F766E;
        --accent-soft: #F0FDFA;
        --ink: #0F172A;
        --ink-muted: #475569;
        --border: #E2E8F0;
        --surface: #F8FAFC;
    }

    /* Titre principal + sous-titre : un peu d'air, séparation nette avec les onglets. */
    h1, h2, h3 {
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] [data-testid="stHeadingWithActionElements"] h1 {
        margin-bottom: 0.1rem;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--ink-muted);
    }

    /* Barre d'onglets : libellés plus grands et plus lisibles, transition douce
       au survol/à la sélection au lieu du style par défaut assez discret. */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    [data-testid="stTab"] {
        font-size: 1rem;
        font-weight: 600;
        color: var(--ink-muted);
        padding: 0.75rem 0.25rem;
        transition: color 0.15s ease;
    }
    [data-testid="stTab"]:hover {
        color: var(--accent);
    }
    [data-testid="stTab"][aria-selected="true"] {
        color: var(--accent);
    }

    /* Barre latérale : léger fond distinct du contenu principal + titres compacts. */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
        background-color: var(--surface);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 1.05rem;
    }

    /* Cartes (st.container(border=True)) : ombre légère et coins arrondis au lieu
       du simple filet gris par défaut, pour détacher visuellement chaque section
       du résultat (aperçu, surface, correction manuelle, etc.). */
    [data-testid="stVerticalBlock"][data-test-scroll-behavior] {
        border-radius: 12px;
        border-color: var(--border);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 6px rgba(15, 23, 42, 0.04);
    }

    /* Statistiques (st.metric) présentées comme de petites tuiles plutôt que du
       texte nu, pour qu'elles ressortent dans les cartes de résultats. */
    [data-testid="stMetric"] {
        background-color: var(--accent-soft);
        border: 1px solid #CCFBF1;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--ink-muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: var(--ink);
    }

    /* Zone de dépôt de fichier : bordure en pointillés teintée au lieu du gris
       neutre par défaut, plus cohérente avec le reste de l'appli. */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
        background-color: var(--surface);
        border: 1px dashed #94A3B8;
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
    }

    /* Panneaux dépliants (réglages du nettoyage, etc.) alignés sur le style carte. */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border-color: var(--border);
    }

    /* Messages d'alerte (info/erreur/succès) : coins arrondis cohérents avec le
       reste plutôt que le rectangle un peu carré par défaut. */
    [data-testid="stAlertContainer"] {
        border-radius: 10px;
    }

    .stButton > button, .stDownloadButton > button {
        font-weight: 600;
        border-radius: 8px;
    }

    /* Bulles de chat façon ChatGPT dans l'onglet Assistant IA :
       message utilisateur aligné à droite en teal, assistant à gauche en gris clair. */
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
        background-color: var(--accent);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 0.6rem 1rem;
        margin-left: auto;
        max-width: 75%;
    }
    [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {
        background-color: #F1F5F9;
        border-radius: 16px;
        padding: 0.6rem 1rem;
        max-width: 75%;
    }

    /* Libellés de boutons sur une seule ligne, tronqués proprement (utile pour
       les titres de conversation longs dans la liste de l'Assistant IA). */
    .stButton > button p {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(run_name, device)
    return model, img_size, device


def read_image(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_mask(uploaded_file, shape_hw):
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def to_pil(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def extract_stroke_mask(canvas_rgba, target_rgb, tol=60):
    """Isole les pixels dessinés (alpha > 0) proches de target_rgb sur le calque du canvas."""
    rgb = canvas_rgba[..., :3].astype(int)
    alpha = canvas_rgba[..., 3]
    dist = np.abs(rgb - np.array(target_rgb)).sum(axis=2)
    return ((dist < tol) & (alpha > 0)).astype(np.uint8)


def fill_traced_outline(stroke_mask):
    """Remplit l'intérieur du tracé : un simple trait a un bord intérieur et un bord
    extérieur, qui se dessinent tous les deux (effet de ligne dédoublée). Remplir
    l'intérieur du tracé de l'humain ne fait qu'une seule ligne propre — ça ne
    dessine rien de plus que ce qu'il a lui-même tracé."""
    contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(stroke_mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


st.title("🩹 Segmentation automatique de plaie")
st.caption(
    "Détection automatique de la plaie par intelligence artificielle (U-Net) : "
    "masque, contour, surface, et suivi de la cicatrisation dans le temps."
)
st.caption(
    "⚠️ Outil de quantification d'image pour la recherche in vitro (scratch assay), "
    "pas un outil d'aide au diagnostic médical ni un dispositif clinique."
)

with st.sidebar:
    st.header("⚙️ Paramètres")
    run_name = st.text_input(
        "Modèle utilisé", value="unet_resnet34",
        help="Nom du modèle entraîné à charger (correspond à un fichier dans outputs/checkpoints/).",
    )
    pixels_per_cm = st.number_input(
        "Ratio pixels / cm (optionnel)", min_value=0.0, value=0.0, step=0.1,
        help="Laissez à 0 pour n'afficher la surface qu'en pixels². "
             "Renseignez-le pour obtenir directement la surface en cm².",
    )
    threshold = st.slider(
        "Seuil de binarisation", 0.0, 1.0, 0.5, 0.05,
        help="Sensibilité de la détection : un seuil plus bas détecte une zone de plaie "
             "plus large, un seuil plus haut la restreint aux pixels les plus certains.",
    )

    st.divider()
    st.subheader("🧹 Prétraitement (optionnel)")
    preprocess_on = st.checkbox(
        "Nettoyer l'image avant prédiction", value=False,
        help="Débruitage + contraste (filtres classiques OpenCV, sans IA). Change l'image "
             "vue par le modèle : à activer seulement si vous vérifiez que ça améliore "
             "réellement le résultat, pas par défaut.",
    )
    with st.expander("Réglages du nettoyage", expanded=False):
        denoise_on = st.checkbox("Débruitage (filtre bilatéral)", value=True, disabled=not preprocess_on)
        flatten_on = st.checkbox(
            "Égalisation de l'éclairage (flat-field)", value=False, disabled=not preprocess_on,
            help="Désactivée par défaut : sur ce modèle, elle confond souvent le signal de "
                 "la plaie avec une variation d'éclairage et le dégrade (surface faussée, "
                 "plaie parfois coupée en plusieurs morceaux). À tester au cas par cas.",
        )
        contrast_on = st.checkbox("Renforcement du contraste (CLAHE)", value=True, disabled=not preprocess_on)

    st.divider()
    st.caption("Ces réglages s'appliquent aux deux onglets.")

tab_single, tab_kinetics = st.tabs(
    ["🔬 Analyse d'une image", "📈 Suivi de cicatrisation"]
)

with tab_single:
    upload_col1, upload_col2 = st.columns(2)
    image_file = upload_col1.file_uploader("Image à analyser", type=["jpg", "jpeg", "png"])
    gt_file = upload_col2.file_uploader(
        "Masque de référence (optionnel)", type=["jpg", "jpeg", "png"],
        help="Pour comparer la prédiction du modèle à une annotation manuelle (score Dice, IoU, erreur de surface).",
    )

    if image_file is not None:
        try:
            model, img_size, device = get_model(run_name)
        except FileNotFoundError:
            st.error(f"Modèle introuvable pour '{run_name}'. "
                     f"Vérifiez que le fichier outputs/checkpoints/{run_name}_best.pt existe.")
            st.stop()

        image_bgr = read_image(image_file)
        image_for_model = (
            clean_image(image_bgr, denoise_on, flatten_on, contrast_on, True)
            if preprocess_on else image_bgr
        )
        mask = predict_mask(model, image_for_model, img_size, device, threshold=threshold)
        contours = extract_contours(mask)
        area_px = wound_area_px(mask)
        overlay = draw_contour_overlay(image_bgr, contours) if contours else image_bgr

        st.divider()
        with st.container(border=True):
            if preprocess_on:
                col0, col1, col2, col3 = st.columns(4)
                col0.image(cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB),
                           caption="Image nettoyée (entrée du modèle)", use_container_width=True)
                _, cleaned_png = cv2.imencode(".png", image_for_model)
                col0.download_button(
                    "⬇️ Télécharger l'image nettoyée",
                    data=cleaned_png.tobytes(),
                    file_name=f"{os.path.splitext(image_file.name)[0]}_nettoyee.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                col1, col2, col3 = st.columns(3)
            col1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Image originale", use_container_width=True)
            col2.image(mask * 255, caption="Masque prédit", use_container_width=True)
            col3.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Contour détecté", use_container_width=True)

        if len(contours) > 1:
            with st.container(border=True):
                st.subheader("🖱️ Supprimer un fragment du masque")
                st.caption(
                    "Le masque contient plusieurs morceaux séparés. Cliquez sur un morceau "
                    "dans l'image pour l'exclure (ex: un faux positif isolé) ; cliquez à "
                    "nouveau dessus pour le réintégrer."
                )

                original_contours = contours  # tous les morceaux, avant filtrage

                excluded_key = f"excluded_fragments_{image_file.name}"
                click_key = f"fragment_click_{image_file.name}"
                last_click_key = f"last_click_time_{image_file.name}"
                st.session_state.setdefault(excluded_key, set())
                st.session_state.setdefault(last_click_key, None)

                pending_click = st.session_state.get(click_key)
                if pending_click is not None and pending_click["unix_time"] != st.session_state[last_click_key]:
                    st.session_state[last_click_key] = pending_click["unix_time"]
                    h0, w0 = image_bgr.shape[:2]
                    scale_x = w0 / pending_click["width"]
                    scale_y = h0 / pending_click["height"]
                    x_full = pending_click["x"] * scale_x
                    y_full = pending_click["y"] * scale_y
                    for i, c in enumerate(original_contours):
                        if cv2.pointPolygonTest(c, (float(x_full), float(y_full)), False) >= 0:
                            st.session_state[excluded_key].symmetric_difference_update({i})
                            break

                if st.session_state[excluded_key]:
                    kept_contours = [c for i, c in enumerate(original_contours) if i not in st.session_state[excluded_key]]
                    mask = np.zeros_like(mask)
                    if kept_contours:
                        cv2.drawContours(mask, kept_contours, -1, 1, thickness=cv2.FILLED)
                    contours = kept_contours
                    area_px = wound_area_px(mask)
                    overlay = draw_contour_overlay(image_bgr, contours) if contours else image_bgr
                    st.caption(f"{len(st.session_state[excluded_key])} morceau(x) exclu(s) "
                               f"— surface recalculée ci-dessous.")

                click_display = image_bgr.copy()
                for i, c in enumerate(original_contours):
                    color = (150, 150, 150) if i in st.session_state[excluded_key] else (0, 0, 255)
                    cv2.drawContours(click_display, [c], -1, color, 3)

                disp_w = min(500, click_display.shape[1])
                streamlit_image_coordinates(
                    cv2.cvtColor(click_display, cv2.COLOR_BGR2RGB),
                    width=disp_w,
                    key=click_key,
                )

        with st.container(border=True):
            st.subheader("📏 Surface")
            c1, c2 = st.columns(2)
            c1.metric("Surface (pixels²)", f"{area_px:,}".replace(",", " "))
            if pixels_per_cm > 0:
                area_cm2 = area_px_to_cm2(area_px, pixels_per_cm)
                c2.metric("Surface (cm²)", f"{area_cm2:.2f}")
            else:
                st.caption(
                    "Les surfaces sont exprimées en pixels². Une calibration spatiale "
                    "(ratio pixels/cm, réglage dans la barre latérale) est nécessaire "
                    "pour obtenir une unité physique (cm², mm²...)."
                )

        resultat_segmentation = {"image": image_file.name, "surface_px2": area_px}
        if pixels_per_cm > 0:
            resultat_segmentation["surface_cm2"] = round(area_cm2, 2)
        st.session_state["dernier_resultat_segmentation"] = resultat_segmentation

        with st.container(border=True):
            st.subheader("✏️ Correction manuelle du contour")
            st.caption(
                "Le contour détecté par l'IA n'est pas toujours parfait. Tracez en vert le "
                "contour de la zone à ajouter, en magenta celui de la zone à retirer — "
                "l'intérieur de votre tracé est rempli automatiquement pour n'avoir qu'une "
                "ligne propre (sinon le trait apparaît en double, bord intérieur et extérieur)."
            )

            h0, w0 = image_bgr.shape[:2]
            scale = min(1.0, 700 / w0)
            canvas_w, canvas_h = int(w0 * scale), int(h0 * scale)

            st.session_state.setdefault("canvas_reset_counter", 0)

            manual_mode = st.checkbox(
                "🗑️ Ignorer le contour de l'IA et tracer la plaie entièrement à la main",
                key="manual_trace_mode",
            )
            base_mask = np.zeros_like(mask) if manual_mode else mask
            canvas_background = image_bgr if manual_mode else overlay

            mode_col, width_col, reset_col = st.columns([2, 1, 1])
            mode_label = mode_col.radio(
                "Mode de dessin", ["➕ Ajouter à la plaie", "➖ Retirer de la plaie"],
                horizontal=True, key="correction_mode",
            )
            stroke_width = width_col.slider("Épaisseur du pinceau", 3, 40, 15, key="correction_stroke_width")
            stroke_color = "#00FF00" if mode_label.startswith("➕") else "#FF00FF"
            with reset_col:
                st.write("")
                if st.button("🔄 Réinitialiser le dessin", use_container_width=True):
                    st.session_state.canvas_reset_counter += 1

            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=to_pil(canvas_background).resize((canvas_w, canvas_h)),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="freedraw",
                key=f"canvas_{image_file.name}_{manual_mode}_{st.session_state.canvas_reset_counter}",
            )

            has_correction = canvas_result.image_data is not None and bool(
                (canvas_result.image_data[..., 3] > 0).any()
            )

            if manual_mode and not has_correction:
                st.caption("Tracez le contour de la plaie en vert pour calculer sa surface.")
            elif has_correction:
                add_small = fill_traced_outline(extract_stroke_mask(canvas_result.image_data, (0, 255, 0)))
                remove_small = fill_traced_outline(extract_stroke_mask(canvas_result.image_data, (255, 0, 255)))
                add_mask = cv2.resize(add_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
                remove_mask = cv2.resize(remove_small, (w0, h0), interpolation=cv2.INTER_NEAREST)

                corrected_mask = base_mask.copy()
                corrected_mask[add_mask > 0] = 1
                corrected_mask[remove_mask > 0] = 0

                corrected_contours = extract_contours(corrected_mask)
                corrected_area_px = wound_area_px(corrected_mask)
                corrected_overlay = (
                    draw_contour_overlay(image_bgr, corrected_contours)
                    if corrected_contours else image_bgr
                )

                st.markdown("**Résultat après correction**")
                cc1, cc2 = st.columns([2, 1])
                cc1.image(
                    cv2.cvtColor(corrected_overlay, cv2.COLOR_BGR2RGB),
                    caption="Contour corrigé", use_container_width=True,
                )
                with cc2:
                    diff_px = corrected_area_px - wound_area_px(base_mask)
                    st.metric(
                        "Surface corrigée (px²)", f"{corrected_area_px:,}".replace(",", " "),
                        delta=f"{diff_px:+,}".replace(",", " "),
                    )
                    if pixels_per_cm > 0:
                        st.metric("Surface corrigée (cm²)",
                                  f"{area_px_to_cm2(corrected_area_px, pixels_per_cm):.2f}")
            else:
                st.caption("Aucune correction dessinée pour l'instant — la surface ci-dessus reste celle de l'IA.")

        if gt_file is not None:
            gt_mask = read_mask(gt_file, mask.shape)
            gt_area_px = wound_area_px(gt_mask)

            pred_t = torch.from_numpy(mask.astype(bool)).unsqueeze(0)
            gt_t = torch.from_numpy(gt_mask.astype(bool)).unsqueeze(0)
            tp, fp, fn, _ = confusion_counts(pred_t, gt_t)
            dice = dice_score(tp, fp, fn).item()
            iou = iou_score(tp, fp, fn).item()
            area_error_pct = 100 * abs(area_px - gt_area_px) / max(gt_area_px, 1)

            with st.container(border=True):
                st.subheader("🆚 Comparaison avec le masque de référence")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Dice", f"{dice:.4f}")
                m2.metric("IoU", f"{iou:.4f}")
                m3.metric("Surface référence (px²)", f"{gt_area_px:,}".replace(",", " "))
                m4.metric("Erreur de surface", f"{area_error_pct:.1f} %")

                gt_contours = extract_contours(gt_mask)
                compare = image_bgr.copy()
                if gt_contours:
                    cv2.drawContours(compare, gt_contours, -1, (0, 255, 0), 3)
                if contours:
                    cv2.drawContours(compare, contours, -1, (0, 0, 255), 3)
                st.image(cv2.cvtColor(compare, cv2.COLOR_BGR2RGB),
                          caption="Vert = annotation manuelle · Rouge = prédiction du modèle",
                          use_container_width=False, width=500)
    else:
        st.info("👆 Chargez une image ci-dessus pour lancer la prédiction.")

with tab_kinetics:
    st.caption(
        "Chargez plusieurs photos du même échantillon prises à différents temps "
        "(ex : 0h, 24h, 48h) pour suivre la fermeture de la plaie au cours du temps."
    )

    kinetics_files = st.file_uploader(
        "Images de la série temporelle",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="kinetics_files",
        help="Sélectionnez toutes les photos de la même expérience, quel que soit l'ordre.",
    )

    if not kinetics_files:
        st.info("👆 Chargez au moins 2 images (idéalement 3 ou plus) pour lancer l'analyse.")
    else:
        kinetics_files = sorted(kinetics_files, key=lambda f: f.name)

        with st.container(border=True):
            st.subheader("🕒 Temps associé à chaque image")
            st.caption(
                "Détecté automatiquement depuis le nom du fichier quand c'est possible "
                "(ex : '24h.jpg' → 24h). Corrigez la valeur si besoin."
            )
            cols = st.columns(len(kinetics_files))
            kinetics_times = []
            for col, f in zip(cols, kinetics_files):
                with col:
                    st.caption(f.name)
                    try:
                        default_t = guess_time_h(f.name)
                    except ValueError:
                        default_t = 0.0
                    t = st.number_input("Temps (h)", value=default_t, step=1.0, key=f"kinetics_time_{f.name}")
                    kinetics_times.append(t)

        analyze_clicked = st.button("▶️ Lancer l'analyse", type="primary", use_container_width=True)

        if analyze_clicked:
            try:
                model, img_size, device = get_model(run_name)
            except FileNotFoundError:
                st.error(f"Modèle introuvable pour '{run_name}'. "
                         f"Vérifiez que le fichier outputs/checkpoints/{run_name}_best.pt existe.")
                st.stop()

            raw_by_name = {f.name: read_image(f) for f in kinetics_files}
            images_with_times = [
                (
                    f.name,
                    clean_image(raw_by_name[f.name], denoise_on, flatten_on, contrast_on, True)
                    if preprocess_on else raw_by_name[f.name],
                    t,
                )
                for f, t in zip(kinetics_files, kinetics_times)
            ]
            rows, masks_by_image = compute_kinetics_from_images(
                model, img_size, device, images_with_times,
                threshold=threshold, pixels_per_cm=pixels_per_cm or None,
            )

            # Sauvegardés en session_state (pas seulement affichés) pour que les
            # résultats restent visibles et que le bouton "résumé scientifique"
            # ci-dessous reste utilisable après le rerun qu'il déclenche lui-même.
            st.session_state["kinetics_rows"] = rows
            st.session_state["kinetics_raw_by_name"] = raw_by_name
            st.session_state["kinetics_masks_by_image"] = masks_by_image
            st.session_state["kinetics_csv_path"] = save_csv(rows, "outputs/kinetics")
            st.session_state["kinetics_plot_path"] = save_plot(rows, "outputs/kinetics")
            st.session_state.pop("kinetics_resume", None)
            st.session_state["kinetics_chat_messages"] = []

        if st.session_state.get("kinetics_rows"):
            rows = st.session_state["kinetics_rows"]
            raw_by_name = st.session_state["kinetics_raw_by_name"]
            masks_by_image = st.session_state["kinetics_masks_by_image"]
            csv_path = st.session_state["kinetics_csv_path"]
            plot_path = st.session_state["kinetics_plot_path"]

            st.divider()

            duration_h = rows[-1]["time_h"] - rows[0]["time_h"]
            final_closure = rows[-1]["closure_pct"]
            avg_speed = final_closure / duration_h if duration_h > 0 else None

            st.session_state["dernier_resultat_segmentation"] = {
                "fermeture_finale_pct": final_closure,
                "duree_h": duration_h,
                "vitesse_moyenne_pct_par_h": round(avg_speed, 2) if avg_speed is not None else None,
            }

            s1, s2, s3 = st.columns(3)
            s1.metric("Fermeture finale", f"{final_closure:.1f} %",
                      help="Par rapport à la surface de la plaie au premier temps de la série.")
            s2.metric("Durée observée", f"{duration_h:.0f} h",
                      help="Écart entre le premier et le dernier temps de la série.")
            s3.metric("Vitesse moyenne de fermeture", f"{avg_speed:.2f} %/h" if avg_speed is not None else "—",
                      help="Fermeture finale divisée par la durée observée : une vitesse moyenne "
                           "(points de % par heure) sur toute la période, pas une vitesse "
                           "instantanée ni une valeur mesurée à chaque instant.")
            st.caption(f"📏 Interprétation basée sur {len(rows)} mesure(s) — voir le détail ci-dessous.")

            temps_mesures = ", ".join(f"{row['time_h']:g}h" for row in rows)
            st.caption(
                f"Suivi basé sur {len(rows)} mesure(s) ({temps_mesures}). "
                "Ces points permettent d'observer l'évolution globale de la fermeture, mais "
                "ne suffisent pas à caractériser précisément la dynamique entre deux temps de "
                "mesure (une accélération ou un ralentissement entre deux points ne serait pas "
                "visible sans mesure intermédiaire)."
            )

            column_labels = {
                "time_h": "Temps (h)", "image": "Image", "area_px2": "Surface (px²)",
                "area_cm2": "Surface (cm²)", "closure_pct": "% fermeture",
            }
            display_rows = [{column_labels.get(k, k): v for k, v in row.items()} for row in rows]

            col_table, col_curve = st.columns([1, 1])
            with col_table:
                with st.container(border=True):
                    st.subheader("📋 Tableau des résultats")
                    st.dataframe(display_rows, use_container_width=True, hide_index=True)
                    if not pixels_per_cm:
                        st.caption(
                            "Surfaces en pixels² : sans calibration spatiale (ratio pixels/cm), "
                            "elles ne sont comparables qu'entre images de cette même série (même "
                            "microscope, même grossissement), pas convertibles en unité physique."
                        )
            with col_curve:
                with st.container(border=True):
                    st.subheader("📈 Courbe de cicatrisation")
                    st.pyplot(build_figure(rows))

            with st.container(border=True):
                st.subheader("🖼️ Vérification visuelle des contours détectés")
                image_by_name = raw_by_name
                gallery_cols = st.columns(len(rows))
                for col, row in zip(gallery_cols, rows):
                    image_bgr = image_by_name[row["image"]]
                    mask = masks_by_image[row["image"]]
                    contours = extract_contours(mask)
                    overlay = draw_contour_overlay(image_bgr, contours) if contours else image_bgr
                    col.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption=f"{row['image']} (t={row['time_h']}h)",
                        use_container_width=True,
                    )

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.write(csv_path, arcname="kinetics.csv")
                zf.write(plot_path, arcname="kinetics_curve.png")

            st.download_button(
                "⬇️ Télécharger les résultats (tableau + courbe)",
                data=zip_buffer.getvalue(),
                file_name="kinetics_results.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.caption(f"💾 Également enregistré dans {csv_path} et {plot_path}")

            with st.container(border=True):
                st.subheader("🤖 Résumé scientifique")
                st.caption(
                    "Transforme le tableau de mesures ci-dessus en un paragraphe de "
                    "résultats prêt à réutiliser, appuyé sur la littérature scientifique."
                )
                if st.button("Générer un résumé scientifique", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Génération du résumé (peut prendre 1 à 2 minutes)..."):
                            morceaux, sources = generer_resume_stream(rows)
                            resume = "".join(morceaux)
                        st.write(resume)
                        if sources:
                            st.caption("Sources : " + ", ".join(sources))
                        st.session_state["kinetics_resume"] = {"texte": resume, "sources": sources}
                    except RuntimeError as exc:
                        st.error(str(exc))
                elif st.session_state.get("kinetics_resume"):
                    st.write(st.session_state["kinetics_resume"]["texte"])
                    sources = st.session_state["kinetics_resume"]["sources"]
                    if sources:
                        st.caption("Sources : " + ", ".join(sources))

            with st.container(border=True):
                st.subheader("💬 Poser une question sur ce résultat")
                st.caption(
                    "L'assistant compare ce résultat à la littérature scientifique sur la "
                    "cicatrisation (RAG local via ChromaDB) et répond via un modèle Ollama "
                    "(mistral). Conversation valable pour cette analyse uniquement (non "
                    "conservée d'une session à l'autre)."
                )
                st.caption(
                    "⚠️ Met en contexte un résultat de recherche in vitro à la lumière de la "
                    "littérature scientifique — ne fournit ni diagnostic ni recommandation "
                    "médicale."
                )

                st.session_state.setdefault("kinetics_chat_messages", [])

                for message in st.session_state["kinetics_chat_messages"]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if message.get("sources"):
                            st.caption("Sources : " + ", ".join(message["sources"]))

                question = st.chat_input("Votre question...", key="kinetics_chat_input")

                if question:
                    st.session_state["kinetics_chat_messages"].append(
                        {"role": "user", "content": question}
                    )
                    with st.chat_message("user"):
                        st.write(question)

                    with st.chat_message("assistant"):
                        try:
                            with st.spinner("Recherche dans la littérature..."):
                                morceaux, sources = repondre_stream(
                                    st.session_state["dernier_resultat_segmentation"], question
                                )
                            reponse = st.write_stream(morceaux)
                        except RuntimeError as exc:
                            st.error(str(exc))
                            st.session_state["kinetics_chat_messages"].append(
                                {"role": "assistant", "content": str(exc), "sources": []}
                            )
                        else:
                            if sources:
                                st.caption("Sources : " + ", ".join(sources))
                            st.session_state["kinetics_chat_messages"].append(
                                {"role": "assistant", "content": reponse, "sources": sources}
                            )
