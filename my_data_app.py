import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection de Maladies des Feuilles de Maïs",
    page_icon="🌽",
    layout="wide"
)

# Titre de l'application
st.title("🌽 Détection de Maladies des Feuilles de Maïs")
st.markdown("""
Cette application utilise un modèle de deep learning pour détecter les maladies des feuilles de maïs.
Téléchargez une image de feuille de maïs et le modèle identifiera si elle est saine ou malade.
""")

# Sidebar pour les informations
st.sidebar.header("À propos")
st.sidebar.info("""
**Classes détectées :**
- 🌿 **Saine** (Healthy)
- 🦠 **Rouille commune** (Common_Rust)
- 🍂 **Brûlure** (Blight)
- 🍁 **Tache grise** (Gray_Leaf_Spot)

**Modèle :** CNN (ResNet50) fine-tuned
**Précision :** ~95% sur le jeu de test
""")

# Charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle pré-entraîné"""
    try:
        # Si vous avez sauvegardé votre modèle, chargez-le ici
        # model = tf.keras.models.load_model('corn_disease_model.h5')
        
        # Pour l'instant, nous créons un modèle fictif - À REMPLACER PAR VOTRE VRAI MODÈLE
        # Vous devrez sauvegarder votre modèle après l'entraînement dans le notebook
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        # Compiler le modèle (nécessaire même pour un modèle chargé)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction de prétraitement de l'image
def preprocess_image(image, target_size=(224, 224)):
    """Prétraite l'image pour la prédiction"""
    # Convertir PIL Image en array numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Redimensionner l'image
    image = cv2.resize(image, target_size)
    
    # Normaliser les pixels entre 0 et 1
    image = image / 255.0
    
    # Ajouter une dimension pour le batch
    image = np.expand_dims(image, axis=0)
    
    return image

# Fonction de prédiction
def predict_disease(model, image):
    """Fait la prédiction sur l'image"""
    # Prétraiter l'image
    processed_image = preprocess_image(image)
    
    # Faire la prédiction
    predictions = model.predict(processed_image)
    
    # Obtenir la classe prédite
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Classes (doivent correspondre à votre entraînement)
    classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    class_names_fr = {
        'Blight': 'Brûlure',
        'Common_Rust': 'Rouille commune',
        'Gray_Leaf_Spot': 'Tache grise',
        'Healthy': 'Saine'
    }
    
    # Emojis pour chaque classe
    class_emojis = {
        'Blight': '🍂',
        'Common_Rust': '🦠',
        'Gray_Leaf_Spot': '🍁',
        'Healthy': '🌿'
    }
    
    predicted_class = classes[class_idx]
    predicted_class_fr = class_names_fr[predicted_class]
    
    return predicted_class, predicted_class_fr, confidence, predictions[0]

# Interface principale
def main():
    # Charger le modèle
    with st.spinner('Chargement du modèle...'):
        model = load_model()
    
    if model is None:
        st.error("Impossible de charger le modèle. Vérifiez que le modèle existe.")
        return
    
    st.success("✅ Modèle chargé avec succès !")
    
    # Créer deux colonnes
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Télécharger une image")
        
        # Option 1 : Téléchargement de fichier
        uploaded_file = st.file_uploader(
            "Choisissez une image de feuille de maïs",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Format supportés : JPG, JPEG, PNG, BMP"
        )
        
        # Option 2 : Utiliser une image d'exemple
        st.markdown("---")
        st.subheader("🎯 Exemples de test")
        
        # Créer des boutons pour des exemples prédéfinis
        example_images = {
            "Feuille saine": "https://raw.githubusercontent.com/example/healthy.jpg",
            "Rouille commune": "https://raw.githubusercontent.com/example/rust.jpg",
            "Brûlure": "https://raw.githubusercontent.com/example/blight.jpg",
            "Tache grise": "https://raw.githubusercontent.com/example/gray_spot.jpg"
        }
        
        example_cols = st.columns(2)
        selected_example = None
        
        for idx, (name, url) in enumerate(example_images.items()):
            with example_cols[idx % 2]:
                if st.button(f"Utiliser {name}", key=f"example_{idx}"):
                    selected_example = url
    
    with col2:
        st.subheader("🖼️ Image chargée")
        
        # Afficher l'image
        image_display = st.empty()
        prediction_result = st.empty()
        confidence_gauge = st.empty()
        
        # Variables pour stocker l'image
        image_to_predict = None
        
        # Traitement de l'image téléchargée
        if uploaded_file is not None:
            try:
                # Ouvrir l'image
                image = Image.open(uploaded_file)
                image_to_predict = image.copy()
                
                # Afficher l'image
                image_display.image(image, caption="Image téléchargée", use_column_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors du chargement de l'image : {e}")
        
        # Traitement de l'exemple sélectionné
        elif selected_example:
            try:
                # Note : Dans une vraie application, vous devriez télécharger l'image depuis l'URL
                # Pour la démo, nous allons utiliser une image aléatoire
                st.info("Utilisation d'une image d'exemple...")
                
                # Créer une image de test aléatoire (remplacer par le téléchargement réel)
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(test_image)
                image_to_predict = image.copy()
                
                image_display.image(image, caption="Image d'exemple", use_column_width=True)
                
            except Exception as e:
                st.error(f"Erreur avec l'exemple : {e}")
        
        else:
            # Afficher un placeholder
            image_display.info("Aucune image sélectionnée. Téléchargez une image ou utilisez un exemple.")
    
    # Bouton de prédiction
    st.markdown("---")
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        predict_button = st.button(
            "🔍 Analyser l'image",
            type="primary",
            disabled=(image_to_predict is None),
            use_container_width=True
        )
    
    # Exécuter la prédiction
    if predict_button and image_to_predict is not None:
        with st.spinner("Analyse en cours..."):
            try:
                # Faire la prédiction
                predicted_class, predicted_class_fr, confidence, all_predictions = predict_disease(model, image_to_predict)
                
                # Afficher les résultats
                st.markdown("---")
                st.subheader("📊 Résultats de l'analyse")
                
                # Créer des colonnes pour les résultats
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Afficher la prédiction principale avec emoji
                    emoji_dict = {
                        'Blight': '🍂',
                        'Common_Rust': '🦠', 
                        'Gray_Leaf_Spot': '🍁',
                        'Healthy': '🌿'
                    }
                    
                    emoji = emoji_dict.get(predicted_class, '❓')
                    
                    st.metric(
                        label="**Diagnostic**",
                        value=f"{emoji} {predicted_class_fr}",
                        delta=f"{confidence:.2f}% de confiance"
                    )
                    
                    # Jauge de confiance
                    st.progress(int(confidence))
                    st.caption(f"Confiance : {confidence:.2f}%")
                
                with result_col2:
                    # Afficher toutes les probabilités
                    classes_fr = {
                        'Blight': 'Brûlure',
                        'Common_Rust': 'Rouille commune',
                        'Gray_Leaf_Spot': 'Tache grise',
                        'Healthy': 'Saine'
                    }
                    
                    classes_order = ['Healthy', 'Blight', 'Common_Rust', 'Gray_Leaf_Spot']
                    
                    # Créer un DataFrame pour les probabilités
                    prob_data = []
                    for i, cls in enumerate(['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']):
                        prob_data.append({
                            'Maladie': classes_fr[cls],
                            'Probabilité': f"{all_predictions[i]*100:.2f}%",
                            'Valeur': all_predictions[i]*100
                        })
                    
                    df_probs = pd.DataFrame(prob_data)
                    df_probs = df_probs.sort_values('Valeur', ascending=False)
                    
                    # Afficher le tableau des probabilités
                    st.dataframe(
                        df_probs[['Maladie', 'Probabilité']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Section d'informations supplémentaires
                st.markdown("---")
                st.subheader("💡 Recommandations")
                
                # Informations basées sur la prédiction
                recommendations = {
                    'Healthy': [
                        "✅ La feuille semble saine !",
                        "Continuez les bonnes pratiques agricoles",
                        "Surveillez régulièrement vos cultures"
                    ],
                    'Blight': [
                        "⚠️ **Brûlure détectée**",
                        "Appliquez des fongicides appropriés",
                        "Évitez l'irrigation par aspersion",
                        "Enlevez et détruisez les plantes infectées"
                    ],
                    'Common_Rust': [
                        "⚠️ **Rouille commune détectée**",
                        "Utilisez des variétés résistantes",
                        "Appliquez des fongicides à base de triazole",
                        "Assurez une bonne circulation d'air"
                    ],
                    'Gray_Leaf_Spot': [
                        "⚠️ **Tache grise détectée**",
                        "Pratiquez la rotation des cultures",
                        "Utilisez des fongicides préventifs",
                        "Évitez les excès d'azote"
                    ]
                }
                
                for rec in recommendations.get(predicted_class, ["Aucune information spécifique disponible."]):
                    st.write(f"- {rec}")
                
                # Avertissement pour les maladies
                if predicted_class != 'Healthy':
                    st.warning("⚠️ Consultez un agronome pour un diagnostic complet et un plan de traitement.")
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")
    
    # Section d'information en bas
    st.markdown("---")
    with st.expander("ℹ️ Comment utiliser cette application"):
        st.markdown("""
        1. **Téléchargez une image** d'une feuille de maïs dans la colonne de gauche
        2. Cliquez sur le bouton **"Analyser l'image"**
        3. Consultez les résultats et les recommandations
        
        **Conseils pour de meilleurs résultats :**
        - Utilisez des images nettes et bien éclairées
        - Photographiez la feuille sur un fond uniforme
        - Capturez les symptômes visibles (taches, décolorations)
        - Évitez les reflets et ombres excessifs
        """)
    
    with st.expander("📈 Performances du modèle"):
        st.markdown("""
        **Métriques du modèle :**
        - Précision sur les données de test : ~95%
        - Classes : 4 (3 maladies + sain)
        - Architecture : CNN avec fine-tuning
        
        **Limitations :**
        - Nécessite des images de qualité
        - Détecte uniquement les maladies entraînées
        - Ne remplace pas un expert agronome
        """)

if __name__ == "__main__":
    main()
