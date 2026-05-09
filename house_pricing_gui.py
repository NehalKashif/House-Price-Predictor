import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class HousePricingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("House Pricing Predictor")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        
        # Initialize model components
        self.model = None
        self.area_scaler = None
        self.feature_list = [
            'area_scaled', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad_code', 'guestroom_code', 'basement_code',
            'hotwaterheating_code', 'airconditioning_code',
            'prefarea_code', 'furnishingstatus_code'
        ]
        
        self.label_encoders = {}
        self.train_model()
        
        self.create_gui()
    
    def train_model(self):
        """Train the model using the Housing.csv dataset"""
        try:
            # Load and preprocess data
            dataset = pd.read_csv('Housing.csv')
            
            # Label encode categorical features
            categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                              'airconditioning', 'prefarea', 'furnishingstatus']
            
            for col in categorical_cols:
                le = LabelEncoder()
                dataset[f'{col}_code'] = le.fit_transform(dataset[col])
                self.label_encoders[col] = le
            
            # Remove outliers
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if not c.endswith("_code")]
            
            Q1 = dataset[numeric_cols].quantile(0.25)
            Q3 = dataset[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (dataset[numeric_cols] < (Q1 - 1.5 * IQR)) | (dataset[numeric_cols] > (Q3 + 1.5 * IQR))
            dataset = dataset.loc[~outlier_mask.any(axis=1)].reset_index(drop=True)
            
            # Scale area
            self.area_scaler = StandardScaler()
            dataset['area_scaled'] = self.area_scaler.fit_transform(dataset[['area']])
            
            # Prepare features and target
            X = dataset[self.feature_list]
            y = dataset['price']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            print("Model trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")
    
    def create_gui(self):
        """Create the GUI layout"""
        # Title
        title_label = ttk.Label(self.root, text="House Price Predictor", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create a frame for inputs
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Dictionary to store input fields
        self.inputs = {}
        
        # Numeric input fields
        row = 0
        ttk.Label(main_frame, text="Area (sq ft):", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['area'] = ttk.Entry(main_frame, width=20)
        self.inputs['area'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['area'].insert(0, "6000")
        row += 1
        
        ttk.Label(main_frame, text="Bedrooms:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['bedrooms'] = ttk.Spinbox(main_frame, from_=1, to=10, width=20)
        self.inputs['bedrooms'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['bedrooms'].set(4)
        row += 1
        
        ttk.Label(main_frame, text="Bathrooms:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['bathrooms'] = ttk.Spinbox(main_frame, from_=1, to=10, width=20)
        self.inputs['bathrooms'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['bathrooms'].set(2)
        row += 1
        
        ttk.Label(main_frame, text="Stories:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['stories'] = ttk.Spinbox(main_frame, from_=1, to=5, width=20)
        self.inputs['stories'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['stories'].set(2)
        row += 1
        
        ttk.Label(main_frame, text="Parking Spaces:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['parking'] = ttk.Spinbox(main_frame, from_=0, to=5, width=20)
        self.inputs['parking'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['parking'].set(1)
        row += 1
        
        # Boolean input fields (Yes/No)
        ttk.Label(main_frame, text="Main Road:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['mainroad'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['mainroad'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['mainroad'].set("Yes")
        row += 1
        
        ttk.Label(main_frame, text="Guest Room:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['guestroom'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['guestroom'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['guestroom'].set("No")
        row += 1
        
        ttk.Label(main_frame, text="Basement:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['basement'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['basement'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['basement'].set("Yes")
        row += 1
        
        ttk.Label(main_frame, text="Hot Water Heating:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['hotwaterheating'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['hotwaterheating'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['hotwaterheating'].set("No")
        row += 1
        
        ttk.Label(main_frame, text="Air Conditioning:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['airconditioning'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['airconditioning'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['airconditioning'].set("No")
        row += 1
        
        ttk.Label(main_frame, text="Preferred Area:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['prefarea'] = ttk.Combobox(main_frame, values=["Yes", "No"], width=18, state="readonly")
        self.inputs['prefarea'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['prefarea'].set("No")
        row += 1
        
        ttk.Label(main_frame, text="Furnishing Status:", font=("Arial", 10)).grid(row=row, column=0, sticky="w", pady=5)
        self.inputs['furnishingstatus'] = ttk.Combobox(main_frame, values=["furnished", "semi-furnished", "unfurnished"], width=18, state="readonly")
        self.inputs['furnishingstatus'].grid(row=row, column=1, sticky="ew", pady=5)
        self.inputs['furnishingstatus'].set("furnished")
        row += 1
        
        # Predict button
        predict_button = ttk.Button(main_frame, text="Predict Price", command=self.predict_price)
        predict_button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=20)
        row += 1
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=10)
        result_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.result_label = ttk.Label(result_frame, text="Enter values and click 'Predict Price'", 
                                      font=("Arial", 12), foreground="gray")
        self.result_label.pack()
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
    
    def predict_price(self):
        """Predict the house price based on user input"""
        try:
            # Collect and validate inputs
            area = float(self.inputs['area'].get())
            bedrooms = int(self.inputs['bedrooms'].get())
            bathrooms = int(self.inputs['bathrooms'].get())
            stories = int(self.inputs['stories'].get())
            parking = int(self.inputs['parking'].get())
            
            # Create raw house data
            raw_new_house = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'stories': stories,
                'parking': parking,
                'mainroad': self.inputs['mainroad'].get().lower(),
                'guestroom': self.inputs['guestroom'].get().lower(),
                'basement': self.inputs['basement'].get().lower(),
                'hotwaterheating': self.inputs['hotwaterheating'].get().lower(),
                'airconditioning': self.inputs['airconditioning'].get().lower(),
                'prefarea': self.inputs['prefarea'].get().lower(),
                'furnishingstatus': self.inputs['furnishingstatus'].get()
            }
            
            # Prepare features
            new_house_df = pd.DataFrame([raw_new_house])
            
            # Scale area
            new_house_df['area_scaled'] = self.area_scaler.transform(new_house_df[['area']])
            
            # Encode categorical features
            for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']:
                new_house_df[f'{col}_code'] = self.label_encoders[col].transform(new_house_df[[col]])
            
            # Select features in correct order
            new_house_df = new_house_df[self.feature_list]
            
            # Make prediction
            predicted_price = self.model.predict(new_house_df)[0]
            
            # Update result label
            self.result_label.config(
                text=f"Predicted Price: Rs {predicted_price:,.2f}",
                foreground="green",
                font=("Arial", 14, "bold")
            )
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid values: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HousePricingGUI(root)
    root.mainloop()
