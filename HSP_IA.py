import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError as MSE, MeanAbsoluteError as MAE
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageTk
import joblib
import threading
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Configuration pour supprimer les warnings TensorFlow
tf.get_logger().setLevel('ERROR')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class HSPPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧪 HSP Prediction - Paramètres de Hansen (δD, δP, δH)")
        self.root.geometry("1600x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialisation du prédicteur
        self.predictor = HSPPredictor()
        self.model_loaded = False
        
        # Variables
        self.smiles_var = tk.StringVar()
        self.molecule_image = None
        self.current_smiles = ""
        self.prediction_history = []  # Historique des prédictions
        
        self.setup_ui()
        self.load_model_async()
    
    def setup_ui(self):
        """Configuration de l'interface utilisateur"""
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Titre principal
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🧪 Prédiction des Paramètres de Hansen (δD, δP, δH)",
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        # Sous-titre
        subtitle_label = tk.Label(
            title_frame,
            text="Hansen Solubility Parameters Predictor",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Conteneur principal avec deux colonnes
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Configuration des colonnes
        main_container.columnconfigure(0, weight=1)  # Colonne de gauche (structure)
        main_container.columnconfigure(1, weight=2)  # Colonne de droite (contrôles)
        main_container.rowconfigure(0, weight=1)
        
        # ========== COLONNE GAUCHE: STRUCTURE MOLÉCULAIRE ==========
        left_column = tk.Frame(main_container, bg='#f0f0f0')
        left_column.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Frame pour la structure moléculaire
        mol_frame = tk.LabelFrame(
            left_column,
            text=" Structure Moléculaire ",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=10,
            pady=10
        )
        mol_frame.pack(fill='both', expand=True)
        
        # Canvas pour afficher la structure
        self.mol_canvas = tk.Canvas(
            mol_frame,
            bg='white',
            relief='solid',
            bd=2,
            width=400,
            height=400
        )
        self.mol_canvas.pack(fill='both', expand=True, pady=10)
        
        # Informations sur la molécule
        self.mol_info_frame = tk.Frame(mol_frame, bg='#f0f0f0')
        self.mol_info_frame.pack(fill='x', pady=5)
        
        self.mol_weight_label = tk.Label(
            self.mol_info_frame,
            text="Poids moléculaire: -",
            font=('Arial', 9),
            bg='#f0f0f0',
            anchor='w'
        )
        self.mol_weight_label.pack(fill='x')
        
        self.mol_formula_label = tk.Label(
            self.mol_info_frame,
            text="Formule: -",
            font=('Arial', 9),
            bg='#f0f0f0',
            anchor='w'
        )
        self.mol_formula_label.pack(fill='x')
        
        self.mol_smiles_label = tk.Label(
            self.mol_info_frame,
            text="SMILES: -",
            font=('Arial', 8),
            bg='#f0f0f0',
            anchor='w',
            wraplength=380
        )
        self.mol_smiles_label.pack(fill='x')
        
        # ========== COLONNE DROITE: CONTRÔLES ET RÉSULTATS ==========
        right_column = tk.Frame(main_container, bg='#f0f0f0')
        right_column.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        
        # Status du modèle
        self.status_frame = tk.Frame(right_column, bg='#f0f0f0')
        self.status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="🔄 Chargement du modèle...",
            font=('Arial', 10),
            fg='orange',
            bg='#f0f0f0'
        )
        self.status_label.pack(side='left')
        
        # Frame de saisie SMILES
        input_frame = tk.LabelFrame(
            right_column,
            text=" Saisie du SMILES ",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=15,
            pady=15
        )
        input_frame.pack(fill='x', pady=(0, 10))
        
        # Label et champ SMILES
        tk.Label(
            input_frame,
            text="SMILES de la molécule:",
            font=('Arial', 10, 'bold'),
            bg='#f0f0f0'
        ).pack(anchor='w')
        
        smiles_input_frame = tk.Frame(input_frame, bg='#f0f0f0')
        smiles_input_frame.pack(fill='x', pady=5)
        
        self.smiles_entry = tk.Entry(
            smiles_input_frame,
            textvariable=self.smiles_var,
            font=('Arial', 11),
            relief='solid',
            bd=1,
            bg='white'
        )
        self.smiles_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.smiles_entry.bind('<KeyRelease>', self.validate_smiles)
        self.smiles_entry.bind('<Return>', lambda e: self.predict_hsp())
        
        # Boutons d'action
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        self.validate_btn = tk.Button(
            button_frame,
            text="✓ Valider SMILES",
            command=self.validate_and_display,
            font=('Arial', 9, 'bold'),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=2,
            width=15
        )
        self.validate_btn.pack(side='left', padx=5)
        
        self.predict_btn = tk.Button(
            button_frame,
            text="🎯 Prédire HSP",
            command=self.predict_hsp,
            font=('Arial', 10, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='raised',
            bd=2,
            width=15,
            state='disabled'
        )
        self.predict_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🔄 Effacer",
            command=self.clear_input,
            font=('Arial', 9),
            bg='#e67e22',
            fg='white',
            relief='raised',
            bd=2,
            width=10
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Label de statut SMILES
        self.smiles_status = tk.Label(
            input_frame,
            text="",
            font=('Arial', 9),
            bg='#f0f0f0'
        )
        self.smiles_status.pack(anchor='w')
        
        # Frame pour les résultats
        results_frame = tk.LabelFrame(
            right_column,
            text=" Résultats de la Prédiction ",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            padx=15,
            pady=15
        )
        results_frame.pack(fill='both', expand=True)
        
        # Zone de texte pour les résultats
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            relief='solid',
            bd=1,
            height=15
        )
        self.results_text.pack(fill='both', expand=True)
        self.results_text.config(state='disabled')
        
        # Frame pour les paramètres prédits (affichage graphique)
        params_frame = tk.Frame(results_frame, bg='#f0f0f0', height=80)
        params_frame.pack(fill='x', pady=(10, 0))
        
        # Labels pour les paramètres prédits
        self.dd_label = tk.Label(
            params_frame,
            text="δD = --",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            bd=2,
            padx=10,
            pady=5
        )
        self.dd_label.pack(side='left', expand=True, fill='x', padx=2)
        
        self.dp_label = tk.Label(
            params_frame,
            text="δP = --",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=2,
            padx=10,
            pady=5
        )
        self.dp_label.pack(side='left', expand=True, fill='x', padx=2)
        
        self.dh_label = tk.Label(
            params_frame,
            text="δH = --",
            font=('Arial', 11, 'bold'),
            bg='#2ecc71',
            fg='white',
            relief='raised',
            bd=2,
            padx=10,
            pady=5
        )
        self.dh_label.pack(side='left', expand=True, fill='x', padx=2)
        
        # Barre de statut
        self.status_bar = tk.Label(
            self.root,
            text="Prêt - En attente de saisie SMILES",
            relief='sunken',
            anchor='w',
            font=('Arial', 9)
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def load_model_async(self):
        """Charge le modèle en arrière-plan"""
        def load_task():
            try:
                success = self.predictor.load_model('hsp_predictor')
                if success:
                    self.root.after(0, self.on_model_loaded)
                else:
                    self.root.after(0, self.on_model_error)
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load_task)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Callback quand le modèle est chargé"""
        self.model_loaded = True
        self.status_label.config(text="✅ Modèle HSP chargé avec succès", fg='green')
        self.update_status_bar("Modèle prêt - Entrez un SMILES valide")
    
    def on_model_error(self, error_msg=""):
        """Callback en cas d'erreur de chargement"""
        self.status_label.config(text="❌ Erreur de chargement du modèle", fg='red')
        messagebox.showerror(
            "Erreur",
            f"Impossible de charger le modèle HSP.\n{error_msg}\n\n"
            "Vérifiez la présence des fichiers:\n"
            "- hsp_predictor_hsp_model.h5\n"
            "- hsp_predictor_preprocessors.pkl"
        )
    
    def validate_smiles(self, event=None):
        """Valide le SMILES en temps réel"""
        smiles = self.smiles_var.get().strip()
        
        if not smiles:
            self.smiles_status.config(text="", fg='black')
            self.predict_btn.config(state='disabled')
            return False
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.smiles_status.config(text="❌ SMILES invalide", fg='red')
            self.predict_btn.config(state='disabled')
            return False
        else:
            self.smiles_status.config(text="✅ SMILES valide", fg='green')
            if self.model_loaded:
                self.predict_btn.config(state='normal')
            return True
    
    def validate_and_display(self):
        """Valide le SMILES et affiche la structure"""
        if self.validate_smiles():
            self.display_molecule_structure()
    
    def display_molecule_structure(self):
        """Affiche la structure de la molécule"""
        smiles = self.smiles_var.get().strip()
        
        if not smiles:
            return
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        
        try:
            # Calcul des propriétés
            mol_weight = Descriptors.MolWt(mol)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            
            # Mise à jour des labels
            self.mol_weight_label.config(text=f"Poids moléculaire: {mol_weight:.2f} g/mol")
            self.mol_formula_label.config(text=f"Formule: {formula}")
            self.mol_smiles_label.config(text=f"SMILES: {smiles[:80]}{'...' if len(smiles) > 80 else ''}")
            
            # Génération de l'image
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.SetFontSize(0.8)
            
            opts = drawer.drawOptions()
            opts.useBWAtomPalette()
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Conversion en image PIL
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            img = img.resize((380, 380), Image.Resampling.LANCZOS)
            
            # Conversion pour Tkinter
            self.molecule_image = ImageTk.PhotoImage(img)
            
            # Affichage sur le canvas
            self.mol_canvas.delete("all")
            self.mol_canvas.create_image(200, 200, image=self.molecule_image)
            
            self.current_smiles = smiles
            
        except Exception as e:
            self.mol_canvas.delete("all")
            self.mol_canvas.create_text(200, 200, 
                                       text="Erreur d'affichage", 
                                       font=('Arial', 12), 
                                       fill='red')
            print(f"Erreur d'affichage: {e}")
    
    def predict_hsp(self):
        """Lance la prédiction HSP"""
        if not self.model_loaded:
            messagebox.showerror("Erreur", "Le modèle n'est pas chargé")
            return
        
        smiles = self.smiles_var.get().strip()
        
        if not smiles:
            messagebox.showwarning("Attention", "Veuillez entrer un SMILES")
            return
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            messagebox.showerror("Erreur", "SMILES invalide")
            return
        
        # Désactiver les boutons
        self.predict_btn.config(state='disabled', text="⏳ Calcul en cours...")
        self.update_status_bar("Prédiction en cours...")
        
        # Lancer la prédiction en arrière-plan
        thread = threading.Thread(target=self.run_prediction, args=(smiles,), daemon=True)
        thread.start()
    
    def run_prediction(self, smiles):
        """Exécute la prédiction dans un thread séparé"""
        try:
            prediction = self.predictor.predict(smiles)
            
            if prediction is not None:
                self.root.after(0, lambda: self.display_prediction_result(smiles, prediction))
            else:
                self.root.after(0, self.display_prediction_error)
                
        except Exception as e:
            self.root.after(0, lambda: self.display_prediction_error(str(e)))
        
        finally:
            self.root.after(0, self.reset_buttons)
    
    def display_prediction_result(self, smiles, prediction):
        """Affiche le résultat de la prédiction"""
        dd, dp, dh = prediction
        
        # Mise à jour des labels colorés
        self.dd_label.config(text=f"δD = {dd:.2f}")
        self.dp_label.config(text=f"δP = {dp:.2f}")
        self.dh_label.config(text=f"δH = {dh:.2f}")
        
        # Calcul du paramètre total et du ratio
        total = dd + dp + dh
        dd_ratio = (dd / total) * 100 if total > 0 else 0
        dp_ratio = (dp / total) * 100 if total > 0 else 0
        dh_ratio = (dh / total) * 100 if total > 0 else 0
        
        # Ajout à l'historique
        self.prediction_history.append({
            'smiles': smiles,
            'dd': dd,
            'dp': dp,
            'dh': dh,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Génération du texte de résultat
        result_text = f"""
{'='*60}
🧪 PRÉDICTION HSP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

🔬 MOLÉCULE:
   SMILES: {smiles}

📊 PARAMÈTRES DE HANSEN PRÉDITS:
   ┌─────────────────────────────────────┐
   │  δD (Dispersion)  =  {dd:8.2f} MPa^(1/2)  │
   │  δP (Polaire)     =  {dp:8.2f} MPa^(1/2)  │
   │  δH (Liaison H)   =  {dh:8.2f} MPa^(1/2)  │
   └─────────────────────────────────────┘

📈 ANALYSE:
   • Paramètre total (δt) = {total:.2f} MPa^(1/2)
   • Contribution dispersion: {dd_ratio:.1f}%
   • Contribution polaire:    {dp_ratio:.1f}%
   • Contribution liaison H:  {dh_ratio:.1f}%

💡 INTERPRÉTATION:
   • δD élevé: Bonne solubilité dans solvants apolaires
   • δP élevé: Sensible aux interactions polaires
   • δH élevé: Peut former des liaisons hydrogène

📌 RÉFÉRENCES (valeurs typiques):
   • Eau:           δD=15.5, δP=16.0, δH=42.3
   • Éthanol:       δD=15.8, δP=8.8,  δH=19.4
   • Hexane:        δD=14.9, δP=0.0,  δH=0.0
   • Acétone:       δD=15.5, δP=10.4, δH=7.0

{'='*60}
"""
        
        # Affichage
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, result_text)
        self.results_text.config(state='disabled')
        
        self.update_status_bar(f"Prédiction terminée - δD={dd:.2f}, δP={dp:.2f}, δH={dh:.2f}")
    
    def display_prediction_error(self, error_msg=""):
        """Affiche une erreur de prédiction"""
        error_text = f"""
{'='*60}
❌ ERREUR DE PRÉDICTION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Impossible d'effectuer la prédiction.

{error_msg}

Vérifiez que:
• Le SMILES est valide
• Le modèle est correctement chargé
• Les fichiers sont accessibles
{'='*60}
"""
        
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, error_text)
        self.results_text.config(state='disabled')
        
        messagebox.showerror("Erreur", "Impossible d'effectuer la prédiction")
    
    def reset_buttons(self):
        """Réactive les boutons"""
        self.predict_btn.config(state='normal', text="🎯 Prédire HSP")
    
    def clear_input(self):
        """Efface la saisie et réinitialise l'affichage"""
        self.smiles_var.set("")
        self.smiles_status.config(text="", fg='black')
        self.predict_btn.config(state='disabled')
        
        # Réinitialiser l'affichage de la structure
        self.mol_canvas.delete("all")
        self.mol_canvas.create_text(200, 200, 
                                   text="Entrez un SMILES\npour afficher la structure", 
                                   font=('Arial', 12), 
                                   fill='gray', 
                                   justify='center')
        
        # Réinitialiser les labels
        self.mol_weight_label.config(text="Poids moléculaire: -")
        self.mol_formula_label.config(text="Formule: -")
        self.mol_smiles_label.config(text="SMILES: -")
        
        # Réinitialiser les paramètres
        self.dd_label.config(text="δD = --")
        self.dp_label.config(text="δP = --")
        self.dh_label.config(text="δH = --")
        
        # Effacer les résultats
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')
        
        self.update_status_bar("Interface réinitialisée")
    
    def update_status_bar(self, message):
        """Met à jour la barre de statut"""
        self.status_bar.config(text=f" {message}")


class HSPPredictor:
    """Classe pour la prédiction des paramètres Hansen"""
    
    def __init__(self, fingerprint_bits=2048, fingerprint_radius=2):
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_radius = fingerprint_radius
        self.scaler = None
        self.model = None
        self.is_trained = False
        self.hsp_params = ['δD', 'δP', 'δH']
    
    def smiles_to_fingerprint(self, smiles):
        """Convertit un SMILES en fingerprint moléculaire"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fingerprint_radius, nBits=self.fingerprint_bits
            )
            return np.array(fingerprint)
            
        except Exception as e:
            print(f"Erreur de conversion SMILES: {e}")
            return None
    
    def calculate_descriptors(self, smiles):
        """Calcule des descripteurs supplémentaires"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(7)
            
            descriptors = []
            descriptors.append(Descriptors.MolWt(mol))
            descriptors.append(Descriptors.NumRotatableBonds(mol))
            descriptors.append(Descriptors.NumHDonors(mol))
            descriptors.append(Descriptors.NumHAcceptors(mol))
            descriptors.append(Descriptors.TPSA(mol))
            descriptors.append(Descriptors.MolLogP(mol))
            descriptors.append(Descriptors.NumAromaticRings(mol))
            
            return np.array(descriptors)
        except:
            return np.zeros(7)
    
    def load_model(self, filepath):
        """Charge le modèle pré-entraîné avec gestion des erreurs"""
        try:
            print(f"Chargement du modèle depuis {filepath}...")
            
            # Désactiver les warnings TensorFlow
            tf.get_logger().setLevel('ERROR')
            
            # Chargement du modèle avec gestion des objets personnalisés
            custom_objects = {
                'mse': MeanSquaredError(),
                'mae': MeanAbsoluteError(),
                'loss': MeanSquaredError(),
                'mean_squared_error': MeanSquaredError(),
                'mean_absolute_error': MeanAbsoluteError(),
                'MeanSquaredError': MeanSquaredError,
                'MeanAbsoluteError': MeanAbsoluteError,
            }
            
            # Essayer de charger le modèle
            model_path = f'{filepath}_hsp_model.h5'
            print(f"Tentative de chargement: {model_path}")
            
            self.model = load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Recompiler le modèle
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            print("Modèle chargé avec succès")
            
            # Chargement des préprocesseurs
            preprocessors_path = f'{filepath}_preprocessors.pkl'
            print(f"Chargement des préprocesseurs: {preprocessors_path}")
            
            preprocessors = joblib.load(preprocessors_path)
            self.scaler = preprocessors['scaler']
            self.fingerprint_bits = preprocessors.get('fingerprint_bits', 2048)
            self.fingerprint_radius = preprocessors.get('fingerprint_radius', 2)
            
            self.is_trained = True
            print("✅ Modèle chargé avec succès!")
            return True
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            import traceback
            traceback.print_exc()
            
            # Tentative avec un modèle simplifié pour le debug
            try:
                print("Tentative de chargement en mode simplifié...")
                self.model = load_model(f'{filepath}_hsp_model.h5', compile=False)
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                preprocessors = joblib.load(f'{filepath}_preprocessors.pkl')
                self.scaler = preprocessors['scaler']
                self.is_trained = True
                print("✅ Modèle chargé en mode simplifié!")
                return True
            except:
                print("❌ Échec du chargement simplifié")
                return False
    
    def predict(self, smiles):
        """Prédit les paramètres Hansen pour un SMILES"""
        if not self.is_trained or self.model is None:
            print("Modèle non entraîné")
            return None
        
        # Conversion en fingerprint
        fp = self.smiles_to_fingerprint(smiles)
        if fp is None:
            print("Fingerprint invalide")
            return None
        
        # Calcul des descripteurs
        desc = self.calculate_descriptors(smiles)
        
        # Combinaison des features
        features = np.concatenate([fp, desc]).reshape(1, -1)
        
        try:
            # Prédiction
            prediction_scaled = self.model.predict(features, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)
            
            # Retourne δD, δP, δH
            return prediction[0]
            
        except Exception as e:
            print(f"Erreur de prédiction: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Configuration pour réduire les messages
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Réduit les logs TensorFlow
    
    root = tk.Tk()
    app = HSPPredictorGUI(root)
    root.mainloop()