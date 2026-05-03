"""
app_gradio.py
-------------
Interface Gradio pour la verification d'identite faciale.
FaceNet vs ArcFace - tout en un seul ecran.

Lancement :
    python app_gradio.py
"""

import numpy as np
from PIL import Image
import torch
import cv2
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ── Chargement des modeles ────────────────────────────────────────────────────
print("Chargement FaceNet...")
from facenet_pytorch import InceptionResnetV1, MTCNN
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn   = MTCNN(image_size=160, margin=14, device=device, keep_all=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("FaceNet OK !")

print("Chargement ArcFace...")
from deepface import DeepFace
print("ArcFace OK !")


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embedding_facenet(img_pil):
    if img_pil is None:
        return None
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    face = mtcnn(img_pil)
    if face is None:
        return None
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(face)
    emb = emb / emb.norm(p=2, dim=1, keepdim=True)
    return emb.cpu().numpy().flatten()


def get_embedding_arcface(img_pil):
    if img_pil is None:
        return None
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    face = mtcnn(img_pil)
    if face is None:
        return None
    face_np  = face.permute(1, 2, 0).numpy()
    face_np  = ((face_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    try:
        result = DeepFace.represent(
            img_path=face_bgr,
            model_name='ArcFace',
            detector_backend='skip',
            enforce_detection=False,
            align=False
        )
        emb = np.array(result[0]['embedding'])
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except:
        return None


def sim(e1, e2):
    return float(np.dot(e1, e2)) if e1 is not None and e2 is not None else None


# ── Carte resultat HTML ───────────────────────────────────────────────────────
def carte(score, seuil, nom, eer):
    if score is None:
        return f"""
        <div style="flex:1;padding:1.2rem;border-radius:10px;
                    background:#1a1a2e;border:1px solid #333;text-align:center;">
            <b style="color:#aaa;">{nom}</b><br>
            <span style="color:#f85149;font-size:0.9rem;">Visage non détecté</span>
        </div>"""
    pct = int(score * 100)
    if score >= seuil:
        verdict, couleur = "MÊME PERSONNE", "#3fb950"
    elif score > seuil * 0.8:
        verdict, couleur = "INCERTAIN", "#e3b341"
    else:
        verdict, couleur = "DIFFÉRENTES", "#f85149"
    return f"""
    <div style="flex:1;padding:1.2rem;border-radius:10px;
                background:rgba(0,0,0,0.25);border:1px solid {couleur}50;text-align:center;">
        <div style="font-size:0.8rem;font-weight:700;color:#aaa;margin-bottom:0.5rem;">{nom} · EER {eer}</div>
        <div style="font-size:2.8rem;font-weight:800;color:{couleur};line-height:1;">{pct}%</div>
        <div style="font-size:0.95rem;font-weight:700;color:{couleur};margin-top:0.3rem;">{verdict}</div>
        <div style="font-size:0.7rem;color:#666;margin-top:0.4rem;">score {score:.4f} · seuil {seuil}</div>
    </div>"""


# ── Fonction principale ───────────────────────────────────────────────────────
def verifier(img1, img2):
    """
    Prend deux images (upload ou webcam),
    lance FaceNet ET ArcFace et affiche les deux resultats.
    """
    if img1 is None or img2 is None:
        return "<p style='color:#e3b341;text-align:center;padding:1rem;'>⚠️ Veuillez fournir les deux images.</p>"

    if not isinstance(img1, Image.Image):
        img1 = Image.fromarray(img1)
    if not isinstance(img2, Image.Image):
        img2 = Image.fromarray(img2)

    e1_fn  = get_embedding_facenet(img1)
    e2_fn  = get_embedding_facenet(img2)
    e1_arc = get_embedding_arcface(img1)
    e2_arc = get_embedding_arcface(img2)

    score_fn  = sim(e1_fn,  e2_fn)
    score_arc = sim(e1_arc, e2_arc)

    return f"""
    <div style="display:flex;gap:1rem;margin-top:0.5rem;">
        {carte(score_fn,  0.4012, "FaceNet (VGGFace2)", "2%")}
        {carte(score_arc, 0.3315, "ArcFace (ResNet50)", "8%")}
    </div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
body, .gradio-container { background: #080d14 !important; font-family: 'Segoe UI', sans-serif; }
.gr-button-primary { background: #1558b0 !important; border: none !important; border-radius: 8px !important; }
.gr-button-primary:hover { background: #1f6feb !important; }
.gr-tab-nav { background: #0d1b2a !important; border-radius: 10px; }
"""

# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Face Verification") as demo:

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#0f1f35,#0a1628);
                border:1px solid rgba(56,139,253,0.2);border-radius:14px;
                padding:1.5rem 2rem;margin-bottom:1.2rem;">
        <h1 style="margin:0;font-size:1.7rem;color:#e6edf3;">🔍 Face Verification System</h1>
        <p style="color:#7d8fa8;font-family:monospace;margin:0.3rem 0 0 0;font-size:0.82rem;">
            FaceNet (EER 2%) vs ArcFace (EER 8%) · Dataset LFW · M1 Deep Learning 2025/2026
        </p>
    </div>
    """)

    with gr.Tabs():

        # ── Onglet 1 : Verification (upload + webcam combines) ────────────────
        with gr.Tab("🔍 Vérification"):
            gr.HTML("""
            <div style="background:#0d1b2a;border:1px solid rgba(56,139,253,0.15);
                        border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:1rem;">
                <span style="color:#388bfd;font-weight:700;">Mode d'emploi : </span>
                <span style="color:#c9d1d9;font-size:0.9rem;">
                Importez ou prenez une photo pour chaque case →
                cliquez sur <b>Vérifier</b> → les deux modèles comparent automatiquement.
                </span>
            </div>
            """)

            with gr.Row():
                img1 = gr.Image(
                    label="📷 Image 1 — Référence (upload ou webcam)",
                    sources=["upload", "webcam"],
                    type="pil",
                    height=300
                )
                img2 = gr.Image(
                    label="📷 Image 2 — À vérifier (upload ou webcam)",
                    sources=["upload", "webcam"],
                    type="pil",
                    height=300
                )

            btn = gr.Button("⚡  Vérifier avec FaceNet ET ArcFace", variant="primary", size="lg")
            resultat = gr.HTML()

            btn.click(fn=verifier, inputs=[img1, img2], outputs=resultat)

        # ── Onglet 2 : Resultats LFW ──────────────────────────────────────────
        with gr.Tab("📊 Résultats LFW"):
            gr.HTML("""
            <div style="padding:1rem;">
                <h3 style="color:#388bfd;margin-top:0;">Évaluation sur 200 paires LFW</h3>
                <table style="width:100%;border-collapse:collapse;margin-top:0.8rem;">
                    <tr style="background:#0d1b2a;color:#e6edf3;text-align:left;">
                        <th style="padding:0.8rem;border:1px solid #1e3a5f;">Modèle</th>
                        <th style="padding:0.8rem;border:1px solid #1e3a5f;">EER</th>
                        <th style="padding:0.8rem;border:1px solid #1e3a5f;">AUC</th>
                        <th style="padding:0.8rem;border:1px solid #1e3a5f;">Seuil optimal</th>
                    </tr>
                    <tr style="color:#c9d1d9;">
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;">FaceNet (VGGFace2)</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;color:#3fb950;font-weight:700;">2.00%</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;color:#3fb950;font-weight:700;">0.9877</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;">0.4012</td>
                    </tr>
                    <tr style="color:#c9d1d9;">
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;">ArcFace (ResNet50)</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;color:#e3b341;font-weight:700;">8.00%</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;color:#e3b341;font-weight:700;">0.9583</td>
                        <td style="padding:0.8rem;border:1px solid #1e3a5f;">0.3315</td>
                    </tr>
                </table>
                <p style="color:#7d8fa8;font-size:0.82rem;margin-top:1rem;">
                    EER (Equal Error Rate) : plus bas = meilleur · AUC : plus proche de 1 = meilleur<br>
                    Seuils calculés au point EER sur les 200 paires LFW.
                </p>
            </div>
            """)


# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860)