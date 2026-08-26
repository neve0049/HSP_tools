import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, colorchooser, filedialog
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import os
import sys
import matplotlib
import random

# Forcer le style avec fond blanc
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['grid.color'] = 'gray'

class SolventApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HSP Solvent Application")
        self.root.geometry("1400x800")
        
        # Facteur de pondération pour δD
        self.weight_dd = 4
        
        # Charger la base de données
        self.load_database()
        
        # Initialiser les listes
        self.solvent_list = []  # Liste 1: Solvent
        self.mixture_list = []  # Liste 2: Mixture of solvent
        self.analyte_list = []  # Liste 3: Analyte
        
        # Couleurs pour les points
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Liste des couleurs prédéfinies pour les couleurs aléatoires
        self.random_colors = [
            '#FF0000', '#0000FF', '#008000', '#FF8C00', '#800080',
            '#A52A2A', '#FF69B4', '#808080', '#808000', '#00BFFF',
            '#FF1493', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF00',
            '#FF4500', '#2E8B57', '#8B008B', '#DC143C', '#006400'
        ]
        
        self.setup_ui()
    
    def generate_random_color(self):
        """Génère une couleur aléatoire"""
        return random.choice(self.random_colors)
    
    def load_database(self):
        """Charge la base de données Excel"""
        try:
            self.df = pd.read_excel('HSPDB.xlsx', usecols='A,B,C,D,F')
            self.df.columns = ['Compound', 'δD', 'δP', 'δH', 'SMILE']
            self.df = self.df.dropna(subset=['Compound'])  # Supprimer les lignes vides
        except Exception as e:
            messagebox.showerror("Error", f"Could not load database: {e}")
            self.df = pd.DataFrame(columns=['Compound', 'δD', 'δP', 'δH', 'SMILE'])
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration des poids pour le redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)  # Ligne pour la légende
        
        # Frame gauche (listes)
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame centre (graphique 3D)
        center_frame = ttk.Frame(main_frame, padding="10")
        center_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        # Frame pour la légende (en dessous des listes et à gauche)
        legend_frame = ttk.LabelFrame(main_frame, text="Legend", padding="5")
        legend_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.S), padx=10, pady=5)
        legend_frame.columnconfigure(0, weight=1)
        
        # Frame droite (boutons)
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=2, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Création des trois listes
        self.create_list_frames(left_frame)
        
        # Création du graphique 3D
        self.setup_3d_plot(center_frame)
        
        # Création de la légende
        self.setup_legend(legend_frame)
        
        # Création des boutons de droite
        self.setup_right_buttons(right_frame)
    
    def setup_legend(self, parent):
        """Crée un frame pour afficher la légende"""
        # Créer un canvas avec scrollbar pour la légende
        legend_canvas = tk.Canvas(parent, height=100)
        legend_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=legend_canvas.xview)
        legend_scrollable_frame = ttk.Frame(legend_canvas)
        
        legend_scrollable_frame.bind(
            "<Configure>",
            lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox("all"))
        )
        
        legend_canvas.create_window((0, 0), window=legend_scrollable_frame, anchor="nw")
        legend_canvas.configure(xscrollcommand=legend_scrollbar.set)
        
        legend_canvas.pack(side="top", fill="x", expand=True)
        legend_scrollbar.pack(side="bottom", fill="x")
        
        # Frame pour contenir les éléments de légende
        self.legend_items_frame = ttk.Frame(legend_scrollable_frame)
        self.legend_items_frame.pack(fill="x", expand=True)
    
    def update_legend(self):
        """Met à jour l'affichage de la légende"""
        # Effacer les éléments existants
        for widget in self.legend_items_frame.winfo_children():
            widget.destroy()
        
        # Créer un conteneur pour organiser les éléments de légende
        items_frame = ttk.Frame(self.legend_items_frame)
        items_frame.pack(fill="x", expand=True)
        
        row = 0
        col = 0
        max_cols = 4  # Nombre maximum d'éléments par ligne
        
        # Ajouter les solvants à la légende
        for item in self.solvent_list:
            self.create_legend_item(items_frame, item, "●", row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Ajouter les mélanges à la légende
        for item in self.mixture_list:
            self.create_legend_item(items_frame, item, "■", row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Ajouter les analytes à la légende
        for item in self.analyte_list:
            self.create_legend_item(items_frame, item, "▲", row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Forcer la mise à jour
        self.legend_items_frame.update_idletasks()
    
    def create_legend_item(self, parent, item, symbol, row, col):
        """Crée un élément individuel dans la légende"""
        # Frame pour un élément de légende
        item_frame = ttk.Frame(parent)
        item_frame.grid(row=row, column=col, padx=10, pady=2, sticky="w")
        
        # Symbole coloré
        symbol_label = tk.Label(item_frame, text=symbol, fg=item['color'], 
                               font=('Arial', 14, 'bold'))
        symbol_label.pack(side="left", padx=(0, 5))
        
        # Nom du composé (tronqué si trop long)
        name = item['name']
        if len(name) > 20:
            name = name[:17] + "..."
        
        name_label = tk.Label(item_frame, text=name, font=('Arial', 9))
        name_label.pack(side="left")
        
        # Type de point
        if symbol == "●":
            type_label = tk.Label(item_frame, text="(S)", font=('Arial', 8), fg="gray")
        elif symbol == "■":
            type_label = tk.Label(item_frame, text="(M)", font=('Arial', 8), fg="gray")
        else:
            type_label = tk.Label(item_frame, text="(A)", font=('Arial', 8), fg="gray")
        
        type_label.pack(side="left", padx=(5, 0))
        
    def create_list_frames(self, parent):
        # Configuration des poids pour les listes
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Liste 1: Solvent
        frame1 = ttk.LabelFrame(parent, text="Solvent", padding="5")
        frame1.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.solvent_listbox = tk.Listbox(frame1, height=10)
        self.solvent_listbox.pack(fill=tk.BOTH, expand=True)
        
        btn_frame1 = ttk.Frame(frame1)
        btn_frame1.pack(fill=tk.X)
        
        ttk.Button(btn_frame1, text="Add", 
                  command=lambda: self.add_item('solvent')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame1, text="Remove", 
                  command=lambda: self.remove_item('solvent')).pack(side=tk.LEFT, padx=2)
        
        # Liste 2: Mixture of solvent
        frame2 = ttk.LabelFrame(parent, text="Mixture of solvent", padding="5")
        frame2.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.mixture_listbox = tk.Listbox(frame2, height=10)
        self.mixture_listbox.pack(fill=tk.BOTH, expand=True)
        
        btn_frame2 = ttk.Frame(frame2)
        btn_frame2.pack(fill=tk.X)
        
        ttk.Button(btn_frame2, text="Add Mixture", 
                  command=self.add_mixture).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame2, text="Remove", 
                  command=lambda: self.remove_item('mixture')).pack(side=tk.LEFT, padx=2)
        
        # Liste 3: Analyte
        frame3 = ttk.LabelFrame(parent, text="Analyte", padding="5")
        frame3.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.analyte_listbox = tk.Listbox(frame3, height=10)
        self.analyte_listbox.pack(fill=tk.BOTH, expand=True)
        
        btn_frame3 = ttk.Frame(frame3)
        btn_frame3.pack(fill=tk.X)
        
        ttk.Button(btn_frame3, text="Add", 
                  command=lambda: self.add_item('analyte')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame3, text="Remove", 
                  command=lambda: self.remove_item('analyte')).pack(side=tk.LEFT, padx=2)
        
    def setup_3d_plot(self, parent):
        # Création de la figure 3D avec fond blanc
        self.fig = Figure(figsize=(8, 6), facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Forcer le fond des axes en blanc
        self.ax.set_facecolor('white')
        
        # Forcer les panneaux (les faces du cube 3D) en blanc
        self.ax.xaxis.pane.set_facecolor('white')
        self.ax.yaxis.pane.set_facecolor('white')
        self.ax.zaxis.pane.set_facecolor('white')
        
        # Rendre les panneaux opaques
        self.ax.xaxis.pane.set_alpha(1.0)
        self.ax.yaxis.pane.set_alpha(1.0)
        self.ax.zaxis.pane.set_alpha(1.0)
        
        # Configurer les axes
        self.ax.set_xlabel('δD')
        self.ax.set_ylabel('δP')
        self.ax.set_zlabel('δH')
        
        # Désactiver la légende automatique de matplotlib
        self.ax.legend_ = None
        
        # Canvas pour intégrer dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_right_buttons(self, parent):
        # Boutons de calcul et gestion
        buttons = [
            ("Calculate Distances", self.calculate_distances),
            ("Export Lists", self.export_lists),
            ("Import Lists", self.import_lists),
            ("HSP_IA", self.run_hsp_ia),
            ("Determine experimental HSP", self.run_exp_hsp)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(parent, text=text, command=command)
            btn.pack(fill=tk.X, pady=5)
    
    def add_item(self, list_type):
        # Fenêtre de dialogue pour ajouter un item
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Compound")
        dialog.geometry("500x1000")
        
        ttk.Label(dialog, text="Choose option:").pack(pady=10)
        
        # Choix de la couleur
        color_frame = ttk.Frame(dialog)
        color_frame.pack(pady=10)
        ttk.Label(color_frame, text="Color:").pack(side=tk.LEFT)
        
        # Variable pour stocker la couleur choisie
        selected_color = [None]
        
        color_label = ttk.Label(color_frame, text="Not chosen", foreground="red")
        color_label.pack(side=tk.LEFT)
        
        def generate_default_color():
            """Génère une couleur aléatoire par défaut"""
            color = self.generate_random_color()
            selected_color[0] = color
            color_label.config(text="Random", foreground=color)
        
        def choose_color_local():
            color = colorchooser.askcolor()[1]
            if color:
                selected_color[0] = color
                color_label.config(text="Chosen", foreground=color)
        
        # Bouton pour choisir une couleur personnalisée
        color_btn = ttk.Button(color_frame, text="Choose Color", command=choose_color_local)
        color_btn.pack(side=tk.LEFT, padx=5)
        
        # Bouton pour générer une couleur aléatoire
        random_btn = ttk.Button(color_frame, text="Random Color", command=generate_default_color)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # Appliquer une couleur aléatoire par défaut
        generate_default_color()
        
        # Frame pour choisir dans la base (AVEC RECHERCHE)
        db_frame = ttk.LabelFrame(dialog, text="Choose from database", padding="10")
        db_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Barre de recherche
        search_frame = ttk.Frame(db_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Liste des composés (avec scrollbar)
        list_frame = ttk.Frame(db_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        compound_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        compound_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=compound_listbox.yview)
        
        # Remplir la liste avec tous les composés
        compounds = self.df['Compound'].tolist()
        for compound in compounds:
            compound_listbox.insert(tk.END, compound)
        
        def update_list(*args):
            search_term = search_var.get().lower()
            compound_listbox.delete(0, tk.END)
            for compound in compounds:
                if search_term in compound.lower():
                    compound_listbox.insert(tk.END, compound)
        
        search_var.trace_add('write', update_list)
        
        def on_double_click(event):
            selection = compound_listbox.curselection()
            if selection:
                selected_compound = compound_listbox.get(selection[0])
                self.add_from_database(list_type, selected_compound, selected_color[0], dialog)
        
        compound_listbox.bind('<Double-Button-1>', on_double_click)
        
        btn_frame = ttk.Frame(db_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Selected", 
                  command=lambda: self.add_from_database(
                      list_type, 
                      compound_listbox.get(compound_listbox.curselection()[0]) if compound_listbox.curselection() else None, 
                      selected_color[0], dialog)).pack()
        
        # Frame pour ajouter nouveau composé
        new_frame = ttk.LabelFrame(dialog, text="Add new compound", padding="10")
        new_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(new_frame, text="Name:").pack()
        name_entry = ttk.Entry(new_frame)
        name_entry.pack(fill=tk.X, pady=2)
        
        ttk.Label(new_frame, text="δD:").pack()
        dd_entry = ttk.Entry(new_frame)
        dd_entry.pack(fill=tk.X, pady=2)
        
        ttk.Label(new_frame, text="δP:").pack()
        dp_entry = ttk.Entry(new_frame)
        dp_entry.pack(fill=tk.X, pady=2)
        
        ttk.Label(new_frame, text="δH:").pack()
        dh_entry = ttk.Entry(new_frame)
        dh_entry.pack(fill=tk.X, pady=2)
        
        ttk.Button(new_frame, text="Add New", 
                  command=lambda: self.add_new_compound(
                      list_type, name_entry.get(), dd_entry.get(), 
                      dp_entry.get(), dh_entry.get(), selected_color[0], dialog)).pack(pady=5)
    
    def add_mixture(self):
        # Fenêtre pour créer un mélange
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Mixture")
        dialog.geometry("600x500")
        
        # Choix de la couleur
        color_frame = ttk.Frame(dialog)
        color_frame.pack(pady=10)
        ttk.Label(color_frame, text="Mixture Color:").pack(side=tk.LEFT)
        
        selected_color = [None]
        color_label = ttk.Label(color_frame, text="Not chosen", foreground="red")
        color_label.pack(side=tk.LEFT)
        
        def generate_default_color():
            """Génère une couleur aléatoire par défaut"""
            color = self.generate_random_color()
            selected_color[0] = color
            color_label.config(text="Random", foreground=color)
        
        def choose_color_local():
            color = colorchooser.askcolor()[1]
            if color:
                selected_color[0] = color
                color_label.config(text="Chosen", foreground=color)
        
        color_btn = ttk.Button(color_frame, text="Choose Color", command=choose_color_local)
        color_btn.pack(side=tk.LEFT, padx=5)
        
        random_btn = ttk.Button(color_frame, text="Random Color", command=generate_default_color)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # Appliquer une couleur aléatoire par défaut
        generate_default_color()
        
        # Frame principal avec deux colonnes
        main_mixture_frame = ttk.Frame(dialog)
        main_mixture_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        main_mixture_frame.columnconfigure(0, weight=1)
        main_mixture_frame.columnconfigure(1, weight=1)
        main_mixture_frame.rowconfigure(0, weight=1)
        
        # Frame gauche - Liste des composés disponibles avec recherche
        available_frame = ttk.LabelFrame(main_mixture_frame, text="Available compounds", padding="10")
        available_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Barre de recherche
        search_frame = ttk.Frame(available_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Liste des composés disponibles
        available_listbox = tk.Listbox(available_frame, height=12)
        available_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Remplir la liste
        compounds = self.df['Compound'].tolist()
        for compound in compounds:
            available_listbox.insert(tk.END, compound)
        
        def update_available_list(*args):
            search_term = search_var.get().lower()
            available_listbox.delete(0, tk.END)
            for compound in compounds:
                if search_term in compound.lower():
                    available_listbox.insert(tk.END, compound)
        
        search_var.trace_add('write', update_available_list)
        
        # Frame droite - Composés sélectionnés
        selected_frame = ttk.LabelFrame(main_mixture_frame, text="Selected compounds", padding="10")
        selected_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        mixture_listbox = tk.Listbox(selected_frame, height=12)
        mixture_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les contrôles d'ajout
        control_frame = ttk.Frame(dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Percentage (%):").pack(side=tk.LEFT)
        percent_entry = ttk.Entry(control_frame, width=10)
        percent_entry.pack(side=tk.LEFT, padx=5)
        
        # Dictionnaire pour stocker les composés et leurs pourcentages
        mixture_data = []
        
        def add_to_mixture():
            selection = available_listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a compound")
                return
            
            compound = available_listbox.get(selection[0])
            percent = percent_entry.get()
            
            if not percent:
                messagebox.showerror("Error", "Please enter percentage")
                return
            
            try:
                percent = float(percent)
                if percent <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid percentage")
                return
            
            # Vérifier si le composé est déjà dans le mélange
            existing = [item for item in mixture_data if item['name'] == compound]
            if existing:
                messagebox.showerror("Error", "Compound already in mixture")
                return
            
            # Récupérer les données du composé
            compound_data = self.df[self.df['Compound'] == compound].iloc[0]
            mixture_data.append({
                'name': compound,
                'percent': percent,
                'dd': compound_data['δD'],
                'dp': compound_data['δP'],
                'dh': compound_data['δH']
            })
            
            mixture_listbox.insert(tk.END, f"{compound}: {percent}%")
            percent_entry.delete(0, tk.END)
        
        ttk.Button(control_frame, text="Add to mixture", command=add_to_mixture).pack(side=tk.LEFT, padx=5)
        
        def remove_from_mixture():
            selection = mixture_listbox.curselection()
            if selection:
                index = selection[0]
                mixture_listbox.delete(index)
                del mixture_data[index]
        
        ttk.Button(control_frame, text="Remove", command=remove_from_mixture).pack(side=tk.LEFT, padx=5)
        
        def validate_mixture():
            if not mixture_data:
                messagebox.showerror("Error", "No compounds in mixture")
                return
            
            # Vérifier que la somme des pourcentages est 100
            total_percent = sum(item['percent'] for item in mixture_data)
            if abs(total_percent - 100) > 0.01:
                messagebox.showerror("Error", f"Total percentage must be 100% (current: {total_percent}%)")
                return
            
            if not selected_color[0]:
                messagebox.showerror("Error", "Please choose a color")
                return
            
            # Calculer la moyenne pondérée
            weighted_dd = sum(item['dd'] * item['percent'] / 100 for item in mixture_data)
            weighted_dp = sum(item['dp'] * item['percent'] / 100 for item in mixture_data)
            weighted_dh = sum(item['dh'] * item['percent'] / 100 for item in mixture_data)
            
            # Créer un nom pour le mélange
            mixture_name = " + ".join([f"{item['name']}({item['percent']}%)" for item in mixture_data])
            
            # Ajouter à la liste des mélanges
            self.mixture_list.append({
                'name': mixture_name,
                'dd': weighted_dd,
                'dp': weighted_dp,
                'dh': weighted_dh,
                'color': selected_color[0],
                'components': mixture_data
            })
            
            self.mixture_listbox.insert(tk.END, mixture_name)
            self.update_plot()
            dialog.destroy()
        
        ttk.Button(dialog, text="Validate Mixture", command=validate_mixture).pack(pady=10)
    
    def add_from_database(self, list_type, compound, color, dialog):
        if not compound:
            messagebox.showerror("Error", "Please select a compound")
            return
        
        if not color:
            messagebox.showerror("Error", "Please choose a color")
            return
        
        # Récupérer les données du composé
        compound_data = self.df[self.df['Compound'] == compound].iloc[0]
        
        item_data = {
            'name': compound,
            'dd': compound_data['δD'],
            'dp': compound_data['δP'],
            'dh': compound_data['δH'],
            'color': color
        }
        
        if list_type == 'solvent':
            self.solvent_list.append(item_data)
            self.solvent_listbox.insert(tk.END, compound)
        elif list_type == 'analyte':
            self.analyte_list.append(item_data)
            self.analyte_listbox.insert(tk.END, compound)
        
        self.update_plot()
        dialog.destroy()
    
    def add_new_compound(self, list_type, name, dd, dp, dh, color, dialog):
        if not all([name, dd, dp, dh, color]):
            messagebox.showerror("Error", "Please fill all fields and choose a color")
            return
        
        try:
            dd = float(dd)
            dp = float(dp)
            dh = float(dh)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for parameters")
            return
        
        item_data = {
            'name': name,
            'dd': dd,
            'dp': dp,
            'dh': dh,
            'color': color
        }
        
        if list_type == 'solvent':
            self.solvent_list.append(item_data)
            self.solvent_listbox.insert(tk.END, name)
        elif list_type == 'analyte':
            self.analyte_list.append(item_data)
            self.analyte_listbox.insert(tk.END, name)
        
        self.update_plot()
        dialog.destroy()
    
    def choose_color(self, label):
        color = colorchooser.askcolor()[1]
        if color:
            label.config(text="Chosen", foreground=color)
    
    def remove_item(self, list_type):
        if list_type == 'solvent':
            selection = self.solvent_listbox.curselection()
            if selection:
                index = selection[0]
                self.solvent_listbox.delete(index)
                del self.solvent_list[index]
        elif list_type == 'mixture':
            selection = self.mixture_listbox.curselection()
            if selection:
                index = selection[0]
                self.mixture_listbox.delete(index)
                del self.mixture_list[index]
        elif list_type == 'analyte':
            selection = self.analyte_listbox.curselection()
            if selection:
                index = selection[0]
                self.analyte_listbox.delete(index)
                del self.analyte_list[index]
        
        self.update_plot()
    
    def update_plot(self):
        # Effacer le graphique
        self.ax.clear()
        
        # Forcer le fond en blanc après clear
        self.ax.set_facecolor('white')
        self.ax.xaxis.pane.set_facecolor('white')
        self.ax.yaxis.pane.set_facecolor('white')
        self.ax.zaxis.pane.set_facecolor('white')
        self.ax.xaxis.pane.set_alpha(1.0)
        self.ax.yaxis.pane.set_alpha(1.0)
        self.ax.zaxis.pane.set_alpha(1.0)
        
        # Configurer les axes
        self.ax.set_xlabel('δD')
        self.ax.set_ylabel('δP')
        self.ax.set_zlabel('δH')
        
        # Désactiver la légende automatique
        self.ax.legend_ = None
        
        # Plotter les solvants
        for item in self.solvent_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='o', s=100)
        
        # Plotter les mélanges
        for item in self.mixture_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='s', s=100)
        
        # Plotter les analytes
        for item in self.analyte_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='^', s=100)
        
        # Mettre à jour la légende personnalisée
        self.update_legend()
        
        self.canvas.draw()
    
    def calculate_distances(self):
        if not self.analyte_list:
            messagebox.showerror("Error", "No analytes to calculate distances")
            return
        
        if not self.solvent_list and not self.mixture_list:
            messagebox.showerror("Error", "No solvents or mixtures to calculate distances")
            return
        
        results = []
        
        # Calculer les distances pour chaque analyte
        for analyte in self.analyte_list:
            analyte_name = analyte['name']
            
            # Distances avec les solvants
            for solvent in self.solvent_list:
                # Distance avec pondération du δD par 4
                distance = np.sqrt(
                    self.weight_dd * (solvent['dd'] - analyte['dd'])**2 +
                    (solvent['dp'] - analyte['dp'])**2 +
                    (solvent['dh'] - analyte['dh'])**2
                )
                results.append(f"{analyte_name} - {solvent['name']}: {distance:.2f}")
            
            # Distances avec les mélanges
            for mixture in self.mixture_list:
                # Distance avec pondération du δD par 4
                distance = np.sqrt(
                    self.weight_dd * (mixture['dd'] - analyte['dd'])**2 +
                    (mixture['dp'] - analyte['dp'])**2 +
                    (mixture['dh'] - analyte['dh'])**2
                )
                results.append(f"{analyte_name} - {mixture['name']}: {distance:.2f}")
        
        # Afficher les résultats
        result_window = tk.Toplevel(self.root)
        result_window.title("Distance Results")
        result_window.geometry("400x300")
        
        text_widget = tk.Text(result_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        for result in results:
            text_widget.insert(tk.END, result + "\n")
    
    def export_lists(self):
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                               filetypes=[("Excel files", "*.xlsx")])
        if filename:
            # Créer un DataFrame avec toutes les listes
            data = []
            
            for item in self.solvent_list:
                data.append({
                    'Type': 'Solvent',
                    'Name': item['name'],
                    'δD': item['dd'],
                    'δP': item['dp'],
                    'δH': item['dh'],
                    'Color': item['color']
                })
            
            for item in self.mixture_list:
                data.append({
                    'Type': 'Mixture',
                    'Name': item['name'],
                    'δD': item['dd'],
                    'δP': item['dp'],
                    'δH': item['dh'],
                    'Color': item['color']
                })
            
            for item in self.analyte_list:
                data.append({
                    'Type': 'Analyte',
                    'Name': item['name'],
                    'δD': item['dd'],
                    'δP': item['dp'],
                    'δH': item['dh'],
                    'Color': item['color']
                })
            
            df_export = pd.DataFrame(data)
            df_export.to_excel(filename, index=False)
            messagebox.showinfo("Success", "Lists exported successfully")
    
    def import_lists(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if filename:
            try:
                df_import = pd.read_excel(filename)
                
                # Vider les listes actuelles
                self.solvent_list = []
                self.mixture_list = []
                self.analyte_list = []
                
                self.solvent_listbox.delete(0, tk.END)
                self.mixture_listbox.delete(0, tk.END)
                self.analyte_listbox.delete(0, tk.END)
                
                # Remplir avec les données importées
                for _, row in df_import.iterrows():
                    item_data = {
                        'name': row['Name'],
                        'dd': row['δD'],
                        'dp': row['δP'],
                        'dh': row['δH'],
                        'color': row['Color']
                    }
                    
                    if row['Type'] == 'Solvent':
                        self.solvent_list.append(item_data)
                        self.solvent_listbox.insert(tk.END, row['Name'])
                    elif row['Type'] == 'Mixture':
                        self.mixture_list.append(item_data)
                        self.mixture_listbox.insert(tk.END, row['Name'])
                    elif row['Type'] == 'Analyte':
                        self.analyte_list.append(item_data)
                        self.analyte_listbox.insert(tk.END, row['Name'])
                
                self.update_plot()
                messagebox.showinfo("Success", "Lists imported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not import file: {e}")
    
    def run_hsp_ia(self):
        try:
            subprocess.Popen([sys.executable, "HSP_IA.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Could not run HSP_IA.py: {e}")
    
    def run_exp_hsp(self):
        try:
            subprocess.Popen([sys.executable, "expHSP.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Could not run expHSP.py: {e}")

def main():
    root = tk.Tk()
    app = SolventApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()