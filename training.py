import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class HSPPredictor:
    def __init__(self, fingerprint_bits=2048, fingerprint_radius=2):
        """
        Initialise le prédicteur des paramètres Hansen
        
        Args:
            fingerprint_bits: Nombre de bits pour les fingerprints moléculaires
            fingerprint_radius: Rayon pour les fingerprints Morgan
        """
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_radius = fingerprint_radius
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.hsp_params = ['δD', 'δP', 'δH']  # Les 3 paramètres à prédire
        
    def smiles_to_fingerprint(self, smiles):
        """
        Convertit un SMILES en fingerprint moléculaire
        
        Args:
            smiles: String SMILES
            
        Returns:
            numpy array: Fingerprint moléculaire
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"⚠️ SMILES invalide: {smiles}")
                return np.zeros(self.fingerprint_bits)
            
            # Génération du fingerprint Morgan (circular fingerprint)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                self.fingerprint_radius, 
                nBits=self.fingerprint_bits
            )
            return np.array(fingerprint)
            
        except Exception as e:
            print(f"❌ Erreur avec SMILES {smiles}: {e}")
            return np.zeros(self.fingerprint_bits)
    
    def calculate_additional_descriptors(self, mol):
        """
        Calcule des descripteurs moléculaires supplémentaires
        (optionnel - peut améliorer les prédictions)
        
        Args:
            mol: Objet RDKit Mol
            
        Returns:
            numpy array: Descripteurs calculés
        """
        try:
            descriptors = []
            # Descripteurs qui peuvent être pertinents pour HSP
            descriptors.append(Descriptors.MolWt(mol))  # Poids moléculaire
            descriptors.append(Descriptors.NumRotatableBonds(mol))  # Liaisons rotatives
            descriptors.append(Descriptors.NumHDonors(mol))  # Donneurs de liaisons H
            descriptors.append(Descriptors.NumHAcceptors(mol))  # Accepteurs de liaisons H
            descriptors.append(Descriptors.TPSA(mol))  # Surface polaire topologique
            descriptors.append(Descriptors.MolLogP(mol))  # LogP
            descriptors.append(Descriptors.NumAromaticRings(mol))  # Nbre de cycles aromatiques
            
            return np.array(descriptors)
        except:
            return np.zeros(7)
    
    def load_and_preprocess_data(self, excel_path, sheet_name=0):
        """
        Charge et prétraite les données depuis le fichier Excel
        
        Args:
            excel_path: Chemin vers le fichier Excel
            sheet_name: Nom ou index de la feuille
            
        Returns:
            X_fp: Fingerprints d'entrée
            X_descriptors: Descripteurs supplémentaires (optionnel)
            y_scaled: Paramètres HSP normalisés
            original_data: Données originales pour référence
        """
        print("📂 Chargement des données...")
        
        # Lecture du fichier Excel
        df = pd.read_excel(excel_path)
        print(f"✅ Données chargées: {len(df)} entrées")
        print(f"📊 Colonnes: {df.columns.tolist()}")
        
        # Vérification des colonnes requises
        required_cols = ['Compound', 'δD', 'δP', 'δH', 'SMILE']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Colonnes manquantes: {missing_cols}")
            # Essayons de trouver les colonnes par index ou noms alternatifs
            df.columns = required_cols  # Forcer les noms de colonnes
        
        # Nettoyage des données
        print("🧹 Nettoyage des données...")
        initial_count = len(df)
        
        # Suppression des lignes avec valeurs manquantes
        df_clean = df.dropna(subset=['Compound', 'δD', 'δP', 'δH', 'SMILE'])
        
        # Vérification que les paramètres sont numériques
        for param in ['δD', 'δP', 'δH']:
            df_clean[param] = pd.to_numeric(df_clean[param], errors='coerce')
        
        df_clean = df_clean.dropna(subset=['δD', 'δP', 'δH'])
        
        print(f"✅ Après nettoyage: {len(df_clean)} entrées")
        if initial_count - len(df_clean) > 0:
            print(f"   {initial_count - len(df_clean)} entrées supprimées")
        
        # Statistiques descriptives
        print("\n📈 Statistiques des paramètres:")
        for param in ['δD', 'δP', 'δH']:
            print(f"   {param}: min={df_clean[param].min():.2f}, "
                  f"max={df_clean[param].max():.2f}, "
                  f"moy={df_clean[param].mean():.2f}, "
                  f"std={df_clean[param].std():.2f}")
        
        # Conversion des SMILES en fingerprints
        print("\n🔄 Conversion SMILES → Fingerprints...")
        fingerprints = []
        additional_descriptors = []
        valid_indices = []
        
        for idx, row in df_clean.iterrows():
            fp = self.smiles_to_fingerprint(row['SMILE'])
            
            # Vérifier si le fingerprint n'est pas vide
            if np.sum(fp) > 0:  # Au moins un bit actif
                fingerprints.append(fp)
                
                # Calcul des descripteurs additionnels
                mol = Chem.MolFromSmiles(row['SMILE'])
                if mol is not None:
                    desc = self.calculate_additional_descriptors(mol)
                    additional_descriptors.append(desc)
                    valid_indices.append(idx)
        
        print(f"✅ Fingerprints générés: {len(fingerprints)}")
        
        if len(fingerprints) == 0:
            raise ValueError("Aucun fingerprint valide n'a pu être généré!")
        
        # Création des matrices
        X_fp = np.array(fingerprints)
        X_descriptors = np.array(additional_descriptors) if additional_descriptors else None
        
        # Préparation des targets (δD, δP, δH)
        y = df_clean.loc[valid_indices, ['δD', 'δP', 'δH']].values
        
        print(f"\n📦 Dimensions finales:")
        print(f"   X_fp: {X_fp.shape}")
        if X_descriptors is not None:
            print(f"   X_descriptors: {X_descriptors.shape}")
        print(f"   y: {y.shape}")
        
        return X_fp, X_descriptors, y, df_clean.loc[valid_indices]
    
    def build_model(self, fp_dim, desc_dim=None):
        """
        Construit le modèle de réseau de neurones
        
        Args:
            fp_dim: Dimension des fingerprints
            desc_dim: Dimension des descripteurs additionnels (optionnel)
            
        Returns:
            model: Modèle Keras
        """
        print("\n🏗️ Construction du modèle...")
        
        # Calcul de la dimension d'entrée totale
        input_dim = fp_dim
        if desc_dim:
            input_dim += desc_dim
        
        # Architecture du modèle
        model = Sequential([
            # Couche d'entrée
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Couches cachées
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            
            # Couche de sortie (3 neurones pour δD, δP, δH)
            Dense(3, activation='linear', name='hsp_output')
        ])
        
        # Compilation du modèle
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("✅ Modèle construit:")
        model.summary()
        
        return model
    
    def train(self, excel_path, test_size=0.2, validation_split=0.15, 
              epochs=200, batch_size=32, use_descriptors=True):
        """
        Entraîne le modèle sur les données
        
        Args:
            excel_path: Chemin vers le fichier Excel
            test_size: Proportion pour le test
            validation_split: Proportion pour la validation
            epochs: Nombre d'époques
            batch_size: Taille des batches
            use_descriptors: Utiliser les descripteurs supplémentaires
            
        Returns:
            history: Historique d'entraînement
        """
        try:
            # Chargement des données
            X_fp, X_desc, y, df_clean = self.load_and_preprocess_data(excel_path)
            
            # Normalisation des paramètres HSP
            print("\n📊 Normalisation des données...")
            y_scaled = self.scaler.fit_transform(y)
            
            # Combinaison des features
            if use_descriptors and X_desc is not None:
                X = np.concatenate([X_fp, X_desc], axis=1)
                print(f"✅ Utilisation des fingerprints + descripteurs: {X.shape}")
            else:
                X = X_fp
                print(f"✅ Utilisation des fingerprints seuls: {X.shape}")
            
            # Split des données
            print("\n🔄 Split des données...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_scaled, test_size=test_size, random_state=42
            )
            
            print(f"   Entraînement: {len(X_train)} échantillons")
            print(f"   Test: {len(X_test)} échantillons")
            
            # Construction du modèle
            desc_dim = X_desc.shape[1] if (use_descriptors and X_desc is not None) else None
            self.model = self.build_model(self.fingerprint_bits, desc_dim)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=25,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Entraînement
            print("\n🚀 Début de l'entraînement...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Évaluation sur le test set
            print("\n📝 Évaluation sur le test set:")
            test_results = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Prédictions pour analyse
            y_pred_scaled = self.model.predict(X_test)
            y_pred = self.scaler.inverse_transform(y_pred_scaled)
            y_true = self.scaler.inverse_transform(y_test)
            
            # Calcul des erreurs par paramètre
            print("\n📊 Performance par paramètre:")
            for i, param in enumerate(self.hsp_params):
                mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
                rmse = np.sqrt(np.mean((y_pred[:, i] - y_true[:, i])**2))
                print(f"   {param}: MAE={mae:.3f}, RMSE={rmse:.3f}")
            
            # Performance globale
            mae_global = np.mean(np.abs(y_pred - y_true))
            rmse_global = np.sqrt(np.mean((y_pred - y_true)**2))
            print(f"\n📊 Performance globale:")
            print(f"   MAE moyen: {mae_global:.3f}")
            print(f"   RMSE moyen: {rmse_global:.3f}")
            
            self.is_trained = True
            
            return history, X_test, y_test, y_true, df_clean
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_model(self, filepath):
        """
        Sauvegarde le modèle et les préprocesseurs
        
        Args:
            filepath: Chemin de base pour la sauvegarde
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        # Sauvegarde du modèle Keras
        self.model.save(f'{filepath}_hsp_model.h5')
        
        # Sauvegarde des préprocesseurs et configuration
        joblib.dump({
            'scaler': self.scaler,
            'fingerprint_bits': self.fingerprint_bits,
            'fingerprint_radius': self.fingerprint_radius,
            'hsp_params': self.hsp_params
        }, f'{filepath}_preprocessors.pkl')
        
        print(f"💾 Modèle sauvegardé: {filepath}_hsp_model.h5")
        print(f"💾 Préprocesseurs sauvegardés: {filepath}_preprocessors.pkl")
    
    def predict_from_smiles(self, smiles):
        """
        Prédit les paramètres HSP à partir d'un SMILES
        
        Args:
            smiles: String SMILES
            
        Returns:
            dict: Paramètres prédits (δD, δP, δH)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Conversion en fingerprint
        fp = self.smiles_to_fingerprint(smiles)
        
        # Prédiction
        fp_reshaped = fp.reshape(1, -1)
        pred_scaled = self.model.predict(fp_reshaped, verbose=0)
        pred = self.scaler.inverse_transform(pred_scaled)[0]
        
        return {
            'δD': pred[0],
            'δP': pred[1],
            'δH': pred[2],
            'SMILES': smiles
        }

def plot_training_history(history):
    """Visualise l'historique d'entraînement"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, title="Prédictions vs Réelles"):
    """Visualise les prédictions vs les valeurs réelles"""
    params = ['δD', 'δP', 'δH']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (ax, param) in enumerate(zip(axes, params)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        
        # Ligne parfaite
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parfait')
        
        ax.set_xlabel(f'{param} Réel')
        ax.set_ylabel(f'{param} Prédit')
        ax.set_title(f'{param}: Prédictions vs Réelles')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("🧪 PRÉDICTEUR DES PARAMÈTRES HANSEN (HSP)")
    print("="*60)
    
    # Initialisation du prédicteur
    predictor = HSPPredictor(fingerprint_bits=2048, fingerprint_radius=2)
    
    try:
        # Entraînement du modèle
        print("\n🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
        print("-"*40)
        
        history, X_test, y_test, y_true, df = predictor.train(
            excel_path='HSPDB.xlsx',  # Votre fichier Excel
            test_size=0.2,
            validation_split=0.15,
            epochs=200,
            batch_size=32,
            use_descriptors=True  # Utiliser les descripteurs supplémentaires
        )
        
        # Visualisation
        print("\n📈 GÉNÉRATION DES GRAPHIQUES")
        print("-"*40)
        
        plot_training_history(history)
        
        # Prédictions sur le test set
        y_pred_scaled = predictor.model.predict(X_test)
        y_pred = predictor.scaler.inverse_transform(y_pred_scaled)
        
        plot_predictions_vs_actual(y_true, y_pred)
        
        # Sauvegarde du modèle
        print("\n💾 SAUVEGARDE DU MODÈLE")
        print("-"*40)
        predictor.save_model('hsp_predictor')
        
        # Test avec quelques exemples
        print("\n🧪 TEST DE PRÉDICTION")
        print("-"*40)
        
        # Prenons quelques SMILES du dataset pour test
        test_smiles = df['SMILE'].iloc[:3].tolist()
        for smiles in test_smiles:
            pred = predictor.predict_from_smiles(smiles)
            print(f"\nSMILES: {pred['SMILES']}")
            print(f"   δD = {pred['δD']:.2f}")
            print(f"   δP = {pred['δP']:.2f}")
            print(f"   δH = {pred['δH']:.2f}")
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n❌ Fichier 'HSPDB.xlsx' non trouvé!")
        print("   Veuillez vérifier le chemin du fichier.")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()