# Reconnaissance Faciale pour la Vérification d'Identité
## Comparaison FaceNet vs ArcFace — Dataset LFW

---

## Description

Ce projet implémente et compare deux modèles de deep learning pour la vérification
d'identité faciale :
- **FaceNet** : InceptionResNetV1, pré-entraîné sur VGGFace2, perte triplet (PyTorch)
- **ArcFace** : ResNet50, pré-entraîné sur MS-Celeb-1M, perte angulaire (TensorFlow/DeepFace)

Les deux modèles sont évalués sur le dataset **LFW** (Labeled Faces in the Wild)
selon les métriques FAR, FRR, EER et AUC-ROC.

---

## Résultats obtenus

| Modèle              | EER    | AUC    | 
|---------------------|--------|--------|
| FaceNet (VGGFace2)  | 2.00%  | 0.9877 | 
| ArcFace (ResNet50)  | 8.00%  | 0.9583 | 

---

## Fichiers du projet

```
projet_reconnaissance_faciale.ipynb   # Code principal 
app_gradio.py                         # Interface graphique Gradio
article_IEEE.tex                      # Rapport LaTeX
README.md                           
```

---

## Installation

```bash
# 1. Créer l'environnement conda
conda create -n facereco python=3.10
conda activate facereco

# 2. Installer les dépendances
pip install facenet-pytorch deepface tf-keras gradio opencv-python torch torchvision numpy matplotlib scikit-learn pillow
```

---

## Comment exécuter

### Le notebook (évaluation sur LFW)
Exécuter `projet_reconnaissance_faciale.ipynb` 

### L'interface Gradio
```bash
conda activate facereco
python app_gradio.py
```
Un lien public `https://xxxxx.gradio.live` est généré automatiquement accessible depuis tout types d'appareils.

---

## Dataset

**LFW** (Labeled Faces in the Wild) — téléchargé automatiquement depuis scikit-learn
200 paires utilisées: 100 genuines + 100 imposteurs.
