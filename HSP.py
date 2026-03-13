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

class SolventApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HSP Solvent Application")
        self.root.geometry("1400x800")
        
        # Charger la base de données
        self.load_database()
        
        # Initialiser les listes
        self.solvent_list = []  # Liste 1: Solvent
        self.mixture_list = []  # Liste 2: Mixture of solvent
        self.analyte_list = []  # Liste 3: Analyte
        
        # Couleurs pour les points
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        self.setup_ui()
        
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
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Frame gauche (listes)
        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame centre (graphique 3D)
        center_frame = ttk.Frame(main_frame, padding="10")
        center_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        # Frame droite (boutons)
        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Création des trois listes
        self.create_list_frames(left_frame)
        
        # Création du graphique 3D
        self.setup_3d_plot(center_frame)
        
        # Création des boutons de droite
        self.setup_right_buttons(right_frame)
        
    def create_list_frames(self, parent):
        # Configuration des poids pour les listes
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(1, weight=1)
        
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
        # Création de la figure 3D
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Configuration des axes
        self.ax.set_xlabel('δD')
        self.ax.set_ylabel('δP')
        self.ax.set_zlabel('δH')
        
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
        color_btn = ttk.Button(color_frame, text="Choose Color", 
                              command=lambda: self.choose_color(color_label))
        color_btn.pack(side=tk.LEFT, padx=5)
        color_label = ttk.Label(color_frame, text="Not chosen", foreground="red")
        color_label.pack(side=tk.LEFT)
        
        # Variable pour stocker la couleur choisie
        selected_color = [None]
        
        def choose_color_local():
            color = colorchooser.askcolor()[1]
            if color:
                selected_color[0] = color
                color_label.config(text="Chosen", foreground=color)
        
        color_btn.config(command=choose_color_local)
        
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
        
        # Utiliser une Listbox au lieu d'une Combobox pour une meilleure visualisation
        compound_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        compound_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=compound_listbox.yview)
        
        # Remplir la liste avec tous les composés
        compounds = self.df['Compound'].tolist()
        for compound in compounds:
            compound_listbox.insert(tk.END, compound)
        
        # Fonction de recherche
        def update_list(*args):
            search_term = search_var.get().lower()
            compound_listbox.delete(0, tk.END)
            
            for compound in compounds:
                if search_term in compound.lower():
                    compound_listbox.insert(tk.END, compound)
        
        search_var.trace('w', update_list)
        
        # Double-clic pour sélectionner rapidement
        def on_double_click(event):
            selection = compound_listbox.curselection()
            if selection:
                selected_compound = compound_listbox.get(selection[0])
                self.add_from_database(list_type, selected_compound, selected_color[0], dialog)
        
        compound_listbox.bind('<Double-Button-1>', on_double_click)
        
        # Bouton pour ajouter le composé sélectionné
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
        color_btn = ttk.Button(color_frame, text="Choose Color", 
                              command=lambda: self.choose_color(color_label))
        color_btn.pack(side=tk.LEFT, padx=5)
        color_label = ttk.Label(color_frame, text="Not chosen", foreground="red")
        color_label.pack(side=tk.LEFT)
        
        selected_color = [None]
        
        def choose_color_local():
            color = colorchooser.askcolor()[1]
            if color:
                selected_color[0] = color
                color_label.config(text="Chosen", foreground=color)
        
        color_btn.config(command=choose_color_local)
        
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
        
        # Fonction de recherche
        def update_available_list(*args):
            search_term = search_var.get().lower()
            available_listbox.delete(0, tk.END)
            for compound in compounds:
                if search_term in compound.lower():
                    available_listbox.insert(tk.END, compound)
        
        search_var.trace('w', update_available_list)
        
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
        
        # Configurer les axes
        self.ax.set_xlabel('δD')
        self.ax.set_ylabel('δP')
        self.ax.set_zlabel('δH')
        
        # Plotter les solvants
        for item in self.solvent_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='o', s=100, label=item['name'])
        
        # Plotter les mélanges
        for item in self.mixture_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='s', s=100, label=item['name'])
        
        # Plotter les analytes
        for item in self.analyte_list:
            self.ax.scatter(item['dd'], item['dp'], item['dh'], 
                          c=item['color'], marker='^', s=100, label=item['name'])
        
        # Ajouter une légende si nécessaire
        if self.solvent_list or self.mixture_list or self.analyte_list:
            self.ax.legend()
        
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
                distance = np.sqrt(
                    (solvent['dd'] - analyte['dd'])**2 +
                    (solvent['dp'] - analyte['dp'])**2 +
                    (solvent['dh'] - analyte['dh'])**2
                )
                results.append(f"{analyte_name} - {solvent['name']}: {distance:.2f}")
            
            # Distances avec les mélanges
            for mixture in self.mixture_list:
                distance = np.sqrt(
                    (mixture['dd'] - analyte['dd'])**2 +
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