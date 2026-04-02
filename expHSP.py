import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import threading

@dataclass
class Solvent:
    """Classe représentant un solvant avec ses paramètres HSP"""
    cas: str
    name: str
    dD: float
    dP: float
    dH: float
    vmol: float = 0.0
    radius: float = 0.0
    is_good: bool = True  # True = bon solvant, False = mauvais solvant

@dataclass
class HSPSphere:
    """Classe représentant une sphère HSP"""
    dD: float
    dP: float
    dH: float
    radius: float
    
    def distance_to(self, solvent: Solvent) -> float:
        """Calcule la distance entre le centre de la sphère et un solvant"""
        return math.sqrt(
            4 * (self.dD - solvent.dD)**2 + 
            (self.dP - solvent.dP)**2 + 
            (self.dH - solvent.dH)**2
        )
    
    def red_value(self, solvent: Solvent) -> float:
        """Calcule la valeur RED (Relative Energy Difference)"""
        dist = self.distance_to(solvent)
        return dist / self.radius if self.radius > 0 else float('inf')
    
    def is_inside(self, solvent: Solvent) -> bool:
        """Vérifie si un solvant est à l'intérieur de la sphère"""
        return self.distance_to(solvent) <= self.radius

class HSPSphereOptimizer:
    """Optimisation HSP robuste (centre optimal + rayon automatique)"""

    def __init__(self, good_solvents, bad_solvents):
        self.good_solvents = good_solvents
        self.bad_solvents = bad_solvents

    def distance(self, center, solvent):
        dD, dP, dH = center
        return math.sqrt(
            4 * (dD - solvent.dD)**2 +
            (dP - solvent.dP)**2 +
            (dH - solvent.dH)**2
        )

    def compute_radius(self, center):
        """Rayon = distance max des bons solvants"""
        return max(self.distance(center, s) for s in self.good_solvents)

    def objective(self, center):
        """
        Objectif :
        - minimiser le rayon
        - pénaliser les mauvais solvants inclus
        """
        R = self.compute_radius(center)

        penalty = 0
        for s in self.bad_solvents:
            d = self.distance(center, s)
            if d < R:
                penalty += (R - d)**2  # pénalité quadratique

        return R + 2 * penalty  # pondération ajustable

    def optimize(self, iterations=3000, step=0.1):
        """Descente simple + random perturbation"""

        # initialisation = barycentre des bons solvants
        dD = np.mean([s.dD for s in self.good_solvents])
        dP = np.mean([s.dP for s in self.good_solvents])
        dH = np.mean([s.dH for s in self.good_solvents])

        best_center = np.array([dD, dP, dH])
        best_score = self.objective(best_center)

        for i in range(iterations):
            # exploration aléatoire
            candidate = best_center + np.random.uniform(-step, step, 3)

            score = self.objective(candidate)

            if score < best_score:
                best_score = score
                best_center = candidate

            # réduction progressive du step
            if i % 500 == 0:
                step *= 0.7

        # calcul final
        R = self.compute_radius(best_center)

        return HSPSphere(best_center[0], best_center[1], best_center[2], R)

    def evaluate(self, sphere):
        """Évalue la sphère finale"""
        inside_bad = 0

        for s in self.bad_solvents:
            d = sphere.distance_to(s)
            if d <= sphere.radius:
                inside_bad += 1

        return inside_bad

class HSP3DVisualizer:
    """Visualisateur 3D pour les sphères HSP"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.canvas = None
        
    def create_plot(self, parent_frame, sphere: HSPSphere, 
                    good_solvents: List[Solvent], bad_solvents: List[Solvent]):
        """Crée une visualisation 3D de la sphère et des solvants"""
        
        # Créer une nouvelle figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Définir les limites des axes
        all_solvents = good_solvents + bad_solvents
        if all_solvents:
            dD_values = [s.dD for s in all_solvents] + [sphere.dD - sphere.radius, sphere.dD + sphere.radius]
            dP_values = [s.dP for s in all_solvents] + [sphere.dP - sphere.radius, sphere.dP + sphere.radius]
            dH_values = [s.dH for s in all_solvents] + [sphere.dH - sphere.radius, sphere.dH + sphere.radius]
            
            # Ajouter une marge de 20%
            margin = 0.2
            dD_min, dD_max = min(dD_values), max(dD_values)
            dP_min, dP_max = min(dP_values), max(dP_values)
            dH_min, dH_max = min(dH_values), max(dH_values)
            
            dD_range = dD_max - dD_min
            dP_range = dP_max - dP_min
            dH_range = dH_max - dH_min
            
            self.ax.set_xlim(dD_min - margin * dD_range, dD_max + margin * dD_range)
            self.ax.set_ylim(dP_min - margin * dP_range, dP_max + margin * dP_range)
            self.ax.set_zlim(dH_min - margin * dH_range, dH_max + margin * dH_range)
        else:
            self.ax.set_xlim(sphere.dD - 10, sphere.dD + 10)
            self.ax.set_ylim(sphere.dP - 10, sphere.dP + 10)
            self.ax.set_zlim(sphere.dH - 10, sphere.dH + 10)
        
        # Tracer la sphère (semi-transparente)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        
        # Points de la sphère (transformation pour tenir compte du facteur 4 sur dD)
        x = sphere.dD + (sphere.radius / 2) * np.outer(np.cos(u), np.sin(v))
        y = sphere.dP + sphere.radius * np.outer(np.sin(u), np.sin(v))
        z = sphere.dH + sphere.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_surface(x, y, z, color='cyan', alpha=0.2, linewidth=0)
        
        # Tracer le centre de la sphère (point bleu)
        self.ax.scatter([sphere.dD], [sphere.dP], [sphere.dH], 
                       color='blue', s=100, label='Centre de la sphère', 
                       edgecolors='darkblue', linewidth=2)
        
        # Tracer les solvants
        if good_solvents:
            good_dD = [s.dD for s in good_solvents]
            good_dP = [s.dP for s in good_solvents]
            good_dH = [s.dH for s in good_solvents]
            self.ax.scatter(good_dD, good_dP, good_dH, 
                           color='green', s=80, label='Bons solvants (dans la sphère)',
                           alpha=0.7, edgecolors='darkgreen', linewidth=1)
        
        if bad_solvents:
            bad_dD = [s.dD for s in bad_solvents]
            bad_dP = [s.dP for s in bad_solvents]
            bad_dH = [s.dH for s in bad_solvents]
            self.ax.scatter(bad_dD, bad_dP, bad_dH, 
                           color='red', s=80, label='Mauvais solvants (hors sphère)',
                           alpha=0.7, edgecolors='darkred', linewidth=1)
        
        # Ajouter les étiquettes des axes
        self.ax.set_xlabel('dD (Dispersion)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('dP (Polaire)', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('dH (Liaison hydrogène)', fontsize=12, fontweight='bold')
        
        # Ajouter un titre
        self.ax.set_title('Visualisation 3D des paramètres HSP', fontsize=14, fontweight='bold')
        
        # Ajouter la légende
        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Ajuster la mise en page
        self.fig.tight_layout()
        
        # Créer le canvas pour tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.draw()
        
        return self.canvas
    
    def update_plot(self, sphere: HSPSphere, good_solvents: List[Solvent], bad_solvents: List[Solvent]):
        """Met à jour la visualisation existante"""
        # Effacer le contenu actuel
        self.ax.clear()
        
        # Recréer le plot
        self.create_plot(self.canvas.get_tk_widget().master, sphere, good_solvents, bad_solvents)
        self.canvas.draw()

class HSPCalculatorApp:
    """Application principale de calcul HSP"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Calculateur de paramètres de solubilité HSP - avec visualisation 3D")
        self.root.geometry("1400x900")
        
        # Données
        self.solvents: Dict[str, Solvent] = {}
        self.good_solvents: List[Solvent] = []
        self.bad_solvents: List[Solvent] = []
        self.current_sphere: Optional[HSPSphere] = None
        self.visualizer = HSP3DVisualizer()
        self.plot_frame = None
        
        self.load_default_solvents()
        self.setup_ui()
        
    def load_default_solvents(self):
        """Charge une liste par défaut de solvants (basée sur les données du fichier)"""
        default_solvents = [
            # CAS, Nom, dD, dP, dH, Vmol
            ("67-64-1", "Acetone", 15.5, 10.4, 7.0, 73.8),
            ("75-05-8", "Acetonitrile", 15.3, 18.0, 6.1, 52.9),
            ("628-63-7", "n-Amyl Acetate", 15.8, 3.3, 6.1, 148.0),
            ("71-41-0", "n-Amyl Alcohol", 15.9, 5.9, 13.9, 108.6),
            ("71-43-2", "Benzene", 18.4, 0.0, 2.0, 52.9),
            ("100-51-6", "Benzyl Alcohol", 18.4, 6.3, 13.7, 103.8),
            ("120-51-4", "Benzyl Benzoate", 20.0, 5.1, 5.2, 190.3),
            ("71-36-3", "1-Butanol", 16.0, 5.7, 15.8, 92.0),
            ("78-92-2", "2-Butanol", 15.8, 5.7, 14.5, 92.0),
            ("123-86-4", "n-Butyl Acetate", 15.8, 3.7, 6.3, 132.6),
            ("540-88-5", "t-Butyl Acetate", 15.0, 3.7, 6.0, 134.8),
            ("75-65-0", "t-Butyl Alcohol", 15.2, 5.1, 14.7, 96.0),
            ("136-60-7", "Butyl Benzoate", 18.3, 5.6, 5.5, 178.1),
            ("124-17-4", "Butyl Diglycol Acetate", 16.0, 4.1, 8.2, 208.2),
            ("112-07-2", "Butyl Glycol Acetate", 15.3, 7.5, 6.8, 171.2),
            ("590-01-2", "n-Butyl Propionate", 15.7, 5.5, 5.9, 149.3),
            ("502-44-3", "Caprolactone (Epsilon)", 18.0, 15.0, 7.4, 110.8),
            ("67-66-3", "Chloroform", 17.8, 3.1, 5.7, 80.5),
            ("108-39-4", "m-Cresol", 18.5, 6.5, 13.7, 105.0),
            ("110-82-7", "Cyclohexane", 16.8, 0.0, 0.2, 108.9),
            ("108-93-0", "Cyclohexanol", 17.4, 4.1, 13.5, 105.7),
            ("108-94-1", "Cyclohexanone", 17.8, 8.4, 5.1, 104.2),
            ("108-83-8", "Di-isoButyl Ketone", 16.0, 3.7, 4.1, 177.4),
            ("123-42-2", "Diacetone Alcohol", 15.8, 8.2, 10.8, 124.3),
            ("60-29-7", "Diethyl Ether", 14.5, 2.9, 4.6, 104.7),
            ("112-34-5", "Diethylene Glycol Monobutyl Ether", 16.0, 7.0, 10.6, 170.4),
            ("108-87-2", "Methyl Cyclohexane", 16.0, 0.0, 1.0, 128.2),
            ("67-68-5", "Dimethyl Sulfoxide (DMSO)", 18.4, 16.4, 10.2, 71.3),
            ("123-91-1", "1,4-Dioxane", 17.5, 1.8, 9.0, 85.7),
            ("646-06-0", "1,3-Dioxolane", 18.1, 6.6, 9.3, 69.9),
            ("2396-61-4", "Dipropylene Glycol", 16.5, 10.6, 17.7, 131.8),
            ("112-28-7", "Dipropylene Glycol Methyl Ether", 15.5, 5.7, 11.2, 156.1),
            ("29911-28-2", "Dipropylene Glycol Mono n-Butyl Ether", 15.7, 6.5, 10.0, 211.2),
            ("64-17-5", "Ethanol", 15.8, 8.8, 19.4, 58.6),
            ("141-78-6", "Ethyl Acetate", 15.8, 5.3, 7.2, 98.6),
            ("100-41-4", "Ethyl Benzene", 17.8, 0.6, 1.4, 122.8),
            ("97-64-3", "Ethyl Lactate", 16.0, 7.6, 12.5, 115.0),
            ("96-49-1", "Ethylene Carbonate", 18.0, 21.7, 5.1, 66.0),
            ("107-21-1", "Ethylene Glycol", 17.0, 11.0, 26.0, 55.9),
            ("111-76-2", "Ethylene Glycol Monobutyl Ether", 16.0, 5.1, 12.3, 132.0),
            ("109-86-4", "Ethylene Glycol Monomethyl Ether", 16.0, 8.2, 15.0, 79.3),
            ("96-48-0", "gamma-Butyrolactone (GBL)", 18.0, 16.6, 7.4, 76.5),
            ("931-40-8", "Glycerol Carbonate", 17.9, 25.5, 17.4, 83.2),
            ("142-82-5", "Heptane", 15.3, 0.0, 0.0, 147.0),
            ("110-54-3", "Hexane", 14.9, 0.0, 0.0, 131.4),
            ("78-83-1", "Iso-Butanol", 15.1, 5.7, 15.9, 92.9),
            ("97-85-8", "Iso-Butyl Isobutyrate", 15.1, 2.8, 5.8, 169.8),
            ("123-92-2", "Iso-Pentyl Acetate", 15.3, 3.1, 7.0, 150.2),
            ("123-51-3", "iso-Pentyl Alcohol", 15.8, 5.2, 13.3, 109.3),
            ("108-21-4", "Iso-Propyl Acetate", 14.9, 4.5, 8.2, 117.1),
            ("108-20-3", "Iso-Propyl Ether", 15.1, 3.2, 3.2, 141.8),
            ("78-59-1", "Isophorone", 17.0, 8.0, 5.0, 150.3),
            ("5989-27-5", "d-Limonene", 17.2, 1.8, 4.3, 162.9),
            ("67-56-1", "Methanol", 14.7, 12.3, 22.3, 40.6),
            ("79-20-9", "Methyl Acetate", 15.5, 7.2, 7.6, 79.8),
            ("111-77-3", "Methyl Carbitol", 16.2, 7.8, 12.6, 118.2),
            ("109-86-4", "Methyl Cellosolve", 16.0, 8.2, 15.0, 79.3),
            ("108-87-2", "Methyl Cyclohexane", 16.0, 0.0, 1.0, 128.2),
            ("78-93-3", "Methyl Ethyl Ketone (MEK)", 16.0, 9.0, 5.1, 90.2),
            ("110-12-3", "Methyl iso-Amyl Ketone", 16.0, 5.7, 4.1, 141.3),
            ("108-11-2", "Methyl iso-Butyl Carbinol", 15.4, 3.3, 12.3, 127.2),
            ("108-10-1", "Methyl Iso-Butyl Ketone (MIBK)", 15.3, 6.1, 4.1, 125.8),
            ("112-62-9", "Methyl Oleate", 16.2, 3.8, 4.5, 340.7),
            ("107-87-9", "Methyl Propyl Ketone", 16.0, 7.6, 4.7, 107.3),
            ("872-50-4", "N-Methyl-2-Pyrrolidone (NMP)", 18.0, 12.3, 7.2, 96.6),
            ("75-09-2", "Methylene Chloride", 17.0, 7.3, 7.1, 64.4),
            ("127-19-5", "N,N-Dimethyl Acetamide", 16.8, 11.5, 10.2, 93.0),
            ("68-12-2", "N,N-Dimethyl Formamide (DMF)", 17.4, 13.7, 11.3, 77.4),
            ("108-03-2", "1-Nitropropane", 16.6, 12.3, 5.5, 89.5),
            ("122-99-6", "2-Phenoxy Ethanol", 17.8, 5.7, 14.3, 124.7),
            ("67-63-0", "2-Propanol", 15.8, 6.1, 16.4, 76.9),
            ("71-23-8", "1-Propanol", 16.0, 6.8, 17.4, 75.1),
            ("109-60-4", "n-Propyl Acetate", 15.3, 4.3, 7.6, 115.8),
            ("106-36-5", "n-Propyl Propanoate", 15.5, 5.6, 5.7, 132.5),
            ("108-32-7", "Propylene Carbonate", 20.0, 18.0, 4.1, 85.2),
            ("5131-66-8", "Propylene Glycol Monobutyl Ether", 15.3, 4.5, 9.2, 132.0),
            ("54839-24-6", "Propylene Glycol Monoethyl Ether Acetate", 15.6, 6.3, 7.7, 155.1),
            ("107-98-2", "Propylene Glycol Monomethyl Ether", 15.6, 6.3, 11.6, 98.2),
            ("108-65-6", "Propylene Glycol Monomethyl Ether Acetate", 15.6, 5.6, 9.8, 137.1),
            ("770-35-4", "Propylene Glycol Phenyl Ether", 17.4, 5.3, 11.5, 143.2),
            ("105-46-4", "sec-Butyl Acetate", 15.0, 3.7, 7.6, 134.0),
            ("126-33-0", "Sulfolane (Tetramethylene Sulfone)", 18.0, 18.0, 9.9, 95.3),
            ("109-99-9", "Tetrahydrofuran (THF)", 16.8, 5.7, 8.0, 81.9),
            ("97-99-4", "Tetrahydrofurfuryl Alcohol", 17.8, 8.2, 12.9, 97.4),
            ("108-88-3", "Toluene", 18.0, 1.4, 2.0, 106.6),
            ("1330-20-7", "Xylene", 17.6, 1.0, 3.1, 123.9),
        ]
        
        for cas, name, dD, dP, dH, vmol in default_solvents:
            self.solvents[cas] = Solvent(cas, name, dD, dP, dH, vmol)
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Panneau principal avec deux colonnes
        main_panel = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panneau de gauche (contrôles)
        left_frame = ttk.Frame(main_panel, width=600)
        main_panel.add(left_frame, weight=1)
        
        # Panneau de droite (visualisation 3D)
        right_frame = ttk.Frame(main_panel, width=600)
        main_panel.add(right_frame, weight=1)
        
        # Configuration du panneau de gauche
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        left_frame.rowconfigure(3, weight=1)
        
        # Titre
        title_label = ttk.Label(left_frame, text="Calculateur de paramètres de solubilité HSP", 
                                 font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Frame des paramètres de la sphère
        sphere_frame = ttk.LabelFrame(left_frame, text="Paramètres de la sphère HSP", padding="10")
        sphere_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        sphere_frame.columnconfigure(1, weight=1)
        sphere_frame.columnconfigure(3, weight=1)
        sphere_frame.columnconfigure(5, weight=1)
        sphere_frame.columnconfigure(7, weight=1)
        
        # Labels et entrées pour les paramètres
        ttk.Label(sphere_frame, text="dD:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.dD_var = tk.StringVar(value="17.83")
        self.dD_entry = ttk.Entry(sphere_frame, textvariable=self.dD_var, width=10)
        self.dD_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(sphere_frame, text="dP:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.dP_var = tk.StringVar(value="9.76")
        self.dP_entry = ttk.Entry(sphere_frame, textvariable=self.dP_var, width=10)
        self.dP_entry.grid(row=0, column=3, padx=5, sticky=tk.W)
        
        ttk.Label(sphere_frame, text="dH:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.dH_var = tk.StringVar(value="8.07")
        self.dH_entry = ttk.Entry(sphere_frame, textvariable=self.dH_var, width=10)
        self.dH_entry.grid(row=0, column=5, padx=5, sticky=tk.W)
        
        ttk.Label(sphere_frame, text="Rayon:").grid(row=0, column=6, padx=5, sticky=tk.W)
        self.radius_var = tk.StringVar(value="7.07")
        self.radius_entry = ttk.Entry(sphere_frame, textvariable=self.radius_var, width=10)
        self.radius_entry.grid(row=0, column=7, padx=5, sticky=tk.W)
        
        # Frame pour la sélection des solvants
        selection_frame = ttk.Frame(left_frame)
        selection_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        selection_frame.columnconfigure(0, weight=1)
        selection_frame.columnconfigure(1, weight=1)
        selection_frame.rowconfigure(1, weight=1)
        
        # Liste des solvants disponibles
        available_frame = ttk.LabelFrame(selection_frame, text="Solvants disponibles", padding="5")
        available_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        available_frame.rowconfigure(0, weight=1)
        available_frame.columnconfigure(0, weight=1)
        
        self.available_listbox = tk.Listbox(available_frame, height=15, selectmode=tk.EXTENDED)
        self.available_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scroll_avail = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, command=self.available_listbox.yview)
        scroll_avail.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.available_listbox.config(yscrollcommand=scroll_avail.set)
        
        # Remplir la liste des solvants disponibles
        for solvent in sorted(self.solvents.values(), key=lambda s: s.name):
            self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
        
        # Frame des boutons de sélection
        button_frame = ttk.Frame(selection_frame)
        button_frame.grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="→ Bon", 
                   command=self.move_to_good).grid(row=0, column=0, pady=2, sticky=tk.W+tk.E)
        ttk.Button(button_frame, text="→ Mauvais", 
                   command=self.move_to_bad).grid(row=1, column=0, pady=2, sticky=tk.W+tk.E)
        ttk.Button(button_frame, text="← Retirer", 
                   command=self.move_back).grid(row=2, column=0, pady=2, sticky=tk.W+tk.E)
        
        # Liste des bons solvants
        good_frame = ttk.LabelFrame(selection_frame, text="Bons solvants (dans la sphère)", padding="5")
        good_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        good_frame.rowconfigure(0, weight=1)
        good_frame.columnconfigure(0, weight=1)
        
        self.good_listbox = tk.Listbox(good_frame, height=10, selectmode=tk.EXTENDED)
        self.good_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scroll_good = ttk.Scrollbar(good_frame, orient=tk.VERTICAL, command=self.good_listbox.yview)
        scroll_good.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.good_listbox.config(yscrollcommand=scroll_good.set)
        
        # Liste des mauvais solvants
        bad_frame = ttk.LabelFrame(selection_frame, text="Mauvais solvants (hors sphère)", padding="5")
        bad_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        bad_frame.rowconfigure(0, weight=1)
        bad_frame.columnconfigure(0, weight=1)
        
        self.bad_listbox = tk.Listbox(bad_frame, height=10, selectmode=tk.EXTENDED)
        self.bad_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scroll_bad = ttk.Scrollbar(bad_frame, orient=tk.VERTICAL, command=self.bad_listbox.yview)
        scroll_bad.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.bad_listbox.config(yscrollcommand=scroll_bad.set)
        
        # Frame des boutons d'action
        action_frame = ttk.Frame(left_frame)
        action_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(action_frame, text="Calculer les distances", 
                   command=self.calculate_distances).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="Optimiser la sphère", 
                   command=self.optimize_sphere).grid(row=0, column=1, padx=5)
        ttk.Button(action_frame, text="Mettre à jour le graphique", 
                   command=self.update_visualization).grid(row=0, column=2, padx=5)
        ttk.Button(action_frame, text="Réinitialiser", 
                   command=self.reset).grid(row=0, column=3, padx=5)
        ttk.Button(action_frame, text="Ajouter un solvant", 
                   command=self.add_solvent_dialog).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(action_frame, text="Sauvegarder les résultats", 
                   command=self.save_results).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(action_frame, text="Charger des solvants", 
                   command=self.load_solvents).grid(row=1, column=2, padx=5, pady=5)
        
        # Frame des résultats
        result_frame = ttk.LabelFrame(left_frame, text="Résultats", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        result_frame.columnconfigure(1, weight=1)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8, width=70)
        self.result_text.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        # Labels pour les statistiques
        ttk.Label(result_frame, text="Fit:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.fit_var = tk.StringVar(value="0%")
        ttk.Label(result_frame, textvariable=self.fit_var).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(result_frame, text="Bons solvants:").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.good_count_var = tk.StringVar(value="0")
        ttk.Label(result_frame, textvariable=self.good_count_var).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        
        # Configuration du panneau de droite (visualisation)
        vis_frame = ttk.LabelFrame(right_frame, text="Visualisation 3D", padding="10")
        vis_frame.pack(fill=tk.BOTH, expand=True)
        
        self.plot_frame = ttk.Frame(vis_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Créer un graphique vide initial
        self.current_sphere = HSPSphere(
            float(self.dD_var.get()),
            float(self.dP_var.get()),
            float(self.dH_var.get()),
            float(self.radius_var.get())
        )
        self.update_visualization()
    
    def move_to_good(self):
        """Déplace les solvants sélectionnés vers la liste des bons solvants"""
        selection = self.available_listbox.curselection()
        for i in reversed(selection):
            item = self.available_listbox.get(i)
            name = item.split(" (")[0]
            cas = item.split("(")[1].rstrip(")")
            
            solvent = self.solvents.get(cas)
            if solvent and solvent not in self.good_solvents and solvent not in self.bad_solvents:
                self.good_solvents.append(solvent)
                self.good_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                self.available_listbox.delete(i)
        
        self.update_counts()
        self.update_visualization()
    
    def move_to_bad(self):
        """Déplace les solvants sélectionnés vers la liste des mauvais solvants"""
        selection = self.available_listbox.curselection()
        for i in reversed(selection):
            item = self.available_listbox.get(i)
            name = item.split(" (")[0]
            cas = item.split("(")[1].rstrip(")")
            
            solvent = self.solvents.get(cas)
            if solvent and solvent not in self.good_solvents and solvent not in self.bad_solvents:
                self.bad_solvents.append(solvent)
                self.bad_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                self.available_listbox.delete(i)
        
        self.update_counts()
        self.update_visualization()
    
    def move_back(self):
        """Retire les solvants sélectionnés des listes"""
        # Retirer des bons solvants
        selection = self.good_listbox.curselection()
        for i in reversed(selection):
            item = self.good_listbox.get(i)
            name = item.split(" (")[0]
            cas = item.split("(")[1].rstrip(")")
            
            solvent = self.solvents.get(cas)
            if solvent in self.good_solvents:
                self.good_solvents.remove(solvent)
                self.good_listbox.delete(i)
                self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
        
        # Retirer des mauvais solvants
        selection = self.bad_listbox.curselection()
        for i in reversed(selection):
            item = self.bad_listbox.get(i)
            name = item.split(" (")[0]
            cas = item.split("(")[1].rstrip(")")
            
            solvent = self.solvents.get(cas)
            if solvent in self.bad_solvents:
                self.bad_solvents.remove(solvent)
                self.bad_listbox.delete(i)
                self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
        
        # Trier la liste disponible
        items = list(self.available_listbox.get(0, tk.END))
        items.sort()
        self.available_listbox.delete(0, tk.END)
        for item in items:
            self.available_listbox.insert(tk.END, item)
        
        self.update_counts()
        self.update_visualization()
    
    def update_counts(self):
        """Met à jour les compteurs"""
        self.good_count_var.set(str(len(self.good_solvents)))
    
    def calculate_distances(self):
        """Calcule les distances pour tous les solvants"""
        try:
            dD = float(self.dD_var.get())
            dP = float(self.dP_var.get())
            dH = float(self.dH_var.get())
            radius = float(self.radius_var.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides")
            return
        
        self.current_sphere = HSPSphere(dD, dP, dH, radius)
        
        # Calculer les résultats
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Résultats du calcul des distances:\n")
        self.result_text.insert(tk.END, "=" * 80 + "\n")
        self.result_text.insert(tk.END, f"Centre de la sphère: dD={dD:.2f}, dP={dP:.2f}, dH={dH:.2f}, Rayon={radius:.2f}\n\n")
        
        # Tous les solvants
        all_solvents = self.good_solvents + self.bad_solvents
        if not all_solvents:
            self.result_text.insert(tk.END, "Aucun solvant sélectionné.\n")
            self.update_visualization()
            return
        
        # Tableau des résultats
        self.result_text.insert(tk.END, f"{'Solvant':<30} {'dD':<6} {'dP':<6} {'dH':<6} {'Distance':<10} {'RED':<8} {'Status':<10}\n")
        self.result_text.insert(tk.END, "-" * 80 + "\n")
        
        correct = 0
        for solvent in all_solvents:
            dist = self.current_sphere.distance_to(solvent)
            red = self.current_sphere.red_value(solvent)
            is_inside = dist <= radius
            
            status = "DANS" if is_inside else "HORS"
            
            # Vérifier la correction
            if (is_inside and solvent in self.good_solvents) or (not is_inside and solvent in self.bad_solvents):
                correct += 1
            
            self.result_text.insert(tk.END, f"{solvent.name:<30} {solvent.dD:<6.1f} {solvent.dP:<6.1f} {solvent.dH:<6.1f} "
                                     f"{dist:<10.2f} {red:<8.2f} {status:<10}\n")
        
        # Statistiques
        total = len(all_solvents)
        fit_percent = (correct / total) * 100 if total > 0 else 0
        self.fit_var.set(f"{fit_percent:.1f}%")
        
        self.result_text.insert(tk.END, "\n" + "=" * 80 + "\n")
        self.result_text.insert(tk.END, f"Fit: {correct}/{total} = {fit_percent:.1f}%\n")
        self.result_text.insert(tk.END, f"Bons solvants: {len(self.good_solvents)}\n")
        self.result_text.insert(tk.END, f"Mauvais solvants: {len(self.bad_solvents)}\n")
        
        # Mettre à jour la visualisation
        self.update_visualization()
    
    def optimize_sphere(self):

    	if len(self.good_solvents) < 2:
        	messagebox.showwarning("Attention", "Au moins 2 bons solvants requis")
        	return

    	self.result_text.delete(1.0, tk.END)
    	self.result_text.insert(tk.END, "Optimisation robuste en cours...\n")
    	self.root.update()

    	optimizer = HSPSphereOptimizer(self.good_solvents, self.bad_solvents)
    	sphere = optimizer.optimize()

    	self.current_sphere = sphere

    	self.dD_var.set(f"{sphere.dD:.3f}")
    	self.dP_var.set(f"{sphere.dP:.3f}")
    	self.dH_var.set(f"{sphere.dH:.3f}")
    	self.radius_var.set(f"{sphere.radius:.3f}")

    	# évaluation
    	inside_bad = optimizer.evaluate(sphere)

    	self.result_text.insert(tk.END, f"\nCentre optimal:\n")
    	self.result_text.insert(tk.END, f"dD={sphere.dD:.3f}, dP={sphere.dP:.3f}, dH={sphere.dH:.3f}\n")
    	self.result_text.insert(tk.END, f"Rayon (auto): {sphere.radius:.3f}\n")
    	self.result_text.insert(tk.END, f"Mauvais solvants dans la sphère: {inside_bad}\n")

    	self.calculate_distances()
    
    def update_visualization(self):
        """Met à jour la visualisation 3D"""
        if not self.current_sphere:
            try:
                self.current_sphere = HSPSphere(
                    float(self.dD_var.get()),
                    float(self.dP_var.get()),
                    float(self.dH_var.get()),
                    float(self.radius_var.get())
                )
            except ValueError:
                return
        
        # Effacer le contenu actuel du frame de plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Créer le nouveau plot
        canvas = self.visualizer.create_plot(
            self.plot_frame,
            self.current_sphere,
            self.good_solvents,
            self.bad_solvents
        )
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def reset(self):
        """Réinitialise toutes les listes"""
        self.good_solvents.clear()
        self.bad_solvents.clear()
        
        self.good_listbox.delete(0, tk.END)
        self.bad_listbox.delete(0, tk.END)
        self.available_listbox.delete(0, tk.END)
        
        for solvent in sorted(self.solvents.values(), key=lambda s: s.name):
            self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
        
        self.result_text.delete(1.0, tk.END)
        self.fit_var.set("0%")
        self.good_count_var.set("0")
        
        # Réinitialiser la sphère
        self.current_sphere = HSPSphere(17.83, 9.76, 8.07, 7.07)
        self.dD_var.set("17.83")
        self.dP_var.set("9.76")
        self.dH_var.set("8.07")
        self.radius_var.set("7.07")
        
        self.update_visualization()
    
    def add_solvent_dialog(self):
        """Ouvre une boîte de dialogue pour ajouter un nouveau solvant"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Ajouter un solvant")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Champs de saisie
        ttk.Label(frame, text="Nom:").grid(row=0, column=0, sticky=tk.W, pady=5)
        name_entry = ttk.Entry(frame, width=30)
        name_entry.grid(row=0, column=1, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(frame, text="CAS:").grid(row=1, column=0, sticky=tk.W, pady=5)
        cas_entry = ttk.Entry(frame, width=30)
        cas_entry.grid(row=1, column=1, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(frame, text="dD:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dD_entry = ttk.Entry(frame, width=30)
        dD_entry.grid(row=2, column=1, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(frame, text="dP:").grid(row=3, column=0, sticky=tk.W, pady=5)
        dP_entry = ttk.Entry(frame, width=30)
        dP_entry.grid(row=3, column=1, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(frame, text="dH:").grid(row=4, column=0, sticky=tk.W, pady=5)
        dH_entry = ttk.Entry(frame, width=30)
        dH_entry.grid(row=4, column=1, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(frame, text="Volume molaire:").grid(row=5, column=0, sticky=tk.W, pady=5)
        vmol_entry = ttk.Entry(frame, width=30)
        vmol_entry.grid(row=5, column=1, pady=5, sticky=tk.W+tk.E)
        
        def save_solvent():
            try:
                name = name_entry.get().strip()
                cas = cas_entry.get().strip()
                dD = float(dD_entry.get())
                dP = float(dP_entry.get())
                dH = float(dH_entry.get())
                vmol = float(vmol_entry.get()) if vmol_entry.get().strip() else 0.0
                
                if not name or not cas:
                    messagebox.showerror("Erreur", "Nom et CAS sont requis")
                    return
                
                if cas in self.solvents:
                    if not messagebox.askyesno("Confirmation", f"Le CAS {cas} existe déjà. Voulez-vous le remplacer?"):
                        return
                
                self.solvents[cas] = Solvent(cas, name, dD, dP, dH, vmol)
                
                # Mettre à jour la liste
                self.available_listbox.delete(0, tk.END)
                for solvent in sorted(self.solvents.values(), key=lambda s: s.name):
                    self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                
                dialog.destroy()
                messagebox.showinfo("Succès", f"Solvant {name} ajouté avec succès")
                
            except ValueError as e:
                messagebox.showerror("Erreur", f"Valeurs numériques invalides: {e}")
        
        ttk.Button(frame, text="Ajouter", command=save_solvent).grid(row=6, column=0, columnspan=2, pady=20)
    
    def save_results(self):
        """Sauvegarde les résultats dans un fichier"""
        from tkinter import filedialog
        import json
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if filename:
            data = {
                "sphere": {
                    "dD": float(self.dD_var.get()),
                    "dP": float(self.dP_var.get()),
                    "dH": float(self.dH_var.get()),
                    "radius": float(self.radius_var.get())
                },
                "good_solvents": [
                    {
                        "cas": s.cas,
                        "name": s.name,
                        "dD": s.dD,
                        "dP": s.dP,
                        "dH": s.dH,
                        "vmol": s.vmol
                    }
                    for s in self.good_solvents
                ],
                "bad_solvents": [
                    {
                        "cas": s.cas,
                        "name": s.name,
                        "dD": s.dD,
                        "dP": s.dP,
                        "dH": s.dH,
                        "vmol": s.vmol
                    }
                    for s in self.bad_solvents
                ]
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Succès", f"Résultats sauvegardés dans {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder: {e}")
    
    def load_solvents(self):
        """Charge des solvants depuis un fichier"""
        from tkinter import filedialog
        import json
        
        filename = filedialog.askopenfilename(
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Mettre à jour la sphère
                if "sphere" in data:
                    self.dD_var.set(str(data["sphere"]["dD"]))
                    self.dP_var.set(str(data["sphere"]["dP"]))
                    self.dH_var.set(str(data["sphere"]["dH"]))
                    self.radius_var.set(str(data["sphere"]["radius"]))
                
                # Réinitialiser les listes
                self.good_solvents.clear()
                self.bad_solvents.clear()
                self.good_listbox.delete(0, tk.END)
                self.bad_listbox.delete(0, tk.END)
                
                # Ajouter les bons solvants
                for s_data in data.get("good_solvents", []):
                    solvent = Solvent(
                        s_data["cas"],
                        s_data["name"],
                        s_data["dD"],
                        s_data["dP"],
                        s_data["dH"],
                        s_data.get("vmol", 0.0)
                    )
                    self.solvents[solvent.cas] = solvent
                    self.good_solvents.append(solvent)
                    self.good_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                
                # Ajouter les mauvais solvants
                for s_data in data.get("bad_solvents", []):
                    solvent = Solvent(
                        s_data["cas"],
                        s_data["name"],
                        s_data["dD"],
                        s_data["dP"],
                        s_data["dH"],
                        s_data.get("vmol", 0.0)
                    )
                    self.solvents[solvent.cas] = solvent
                    self.bad_solvents.append(solvent)
                    self.bad_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                
                # Mettre à jour la liste disponible
                self.available_listbox.delete(0, tk.END)
                for solvent in sorted(self.solvents.values(), key=lambda s: s.name):
                    if solvent not in self.good_solvents and solvent not in self.bad_solvents:
                        self.available_listbox.insert(tk.END, f"{solvent.name} ({solvent.cas})")
                
                self.update_counts()
                self.calculate_distances()
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger le fichier: {e}")

def main():
    root = tk.Tk()
    app = HSPCalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()