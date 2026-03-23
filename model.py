import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time

# ======================================================
# NEURAL NETWORK REGRESSOR WITH ENHANCED FEATURES
# ======================================================
class NeuralNetworkScratch:
    def __init__(self, input_size, hidden_size=64, lr=0.005, epochs=500, batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.best_weights = None
        self.training_time = 0

        # Xavier/Glorot initialization
        limit1 = np.sqrt(6 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        limit2 = np.sqrt(6 / (hidden_size + 1))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def fit_with_callback(self, X, y, X_val=None, y_val=None, callback=None):
        """Fit model with progress callback"""
        start_time = time.time()
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Update callback if provided
            if callback:
                callback.update()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                z1 = X_batch @ self.W1 + self.b1
                a1 = self._relu(z1)
                y_pred = a1 @ self.W2 + self.b2
                
                # Loss
                loss = np.mean((y_batch - y_pred) ** 2)
                epoch_loss += loss * len(X_batch)
                
                # Backward pass
                dZ2 = 2 * (y_pred - y_batch) / len(X_batch)
                dW2 = a1.T @ dZ2
                db2 = np.sum(dZ2, axis=0, keepdims=True)
                
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * self._relu_derivative(z1)
                dW1 = X_batch.T @ dZ1
                db1 = np.sum(dZ1, axis=0, keepdims=True)
                
                # Update weights
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            
            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)
            
            # Validation loss
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = np.mean((y_val - val_pred) ** 2)
                self.val_losses.append(val_loss)
                
                # Track best weights
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_weights = {
                        'W1': self.W1.copy(),
                        'b1': self.b1.copy(),
                        'W2': self.W2.copy(),
                        'b2': self.b2.copy()
                    }
            
            # Learning rate decay
            if epoch % 100 == 0 and epoch > 0:
                self.lr *= 0.95
        
        # Restore best weights if validation was used
        if X_val is not None and y_val is not None and self.best_weights:
            self.W1 = self.best_weights['W1']
            self.b1 = self.best_weights['b1']
            self.W2 = self.best_weights['W2']
            self.b2 = self.best_weights['b2']
        
        self.training_time = time.time() - start_time
    
    def fit_with_early_stopping(self, X_train, y_train, X_val, y_val, 
                                callback=None, patience=50, min_delta=1e-4):
        """Fit model with early stopping"""
        start_time = time.time()
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        wait = 0
        
        for epoch in range(self.epochs):
            # Update callback if provided
            if callback:
                callback.update()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                z1 = X_batch @ self.W1 + self.b1
                a1 = self._relu(z1)
                y_pred = a1 @ self.W2 + self.b2
                
                # Loss
                loss = np.mean((y_batch - y_pred) ** 2)
                epoch_loss += loss * len(X_batch)
                
                # Backward pass
                dZ2 = 2 * (y_pred - y_batch) / len(X_batch)
                dW2 = a1.T @ dZ2
                db2 = np.sum(dZ2, axis=0, keepdims=True)
                
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * self._relu_derivative(z1)
                dW1 = X_batch.T @ dZ1
                db1 = np.sum(dZ1, axis=0, keepdims=True)
                
                # Update weights
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            
            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)
            
            # Validation loss
            val_pred = self.predict(X_val)
            val_loss = np.mean((y_val - val_pred) ** 2)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                self.best_weights = {
                    'W1': self.W1.copy(),
                    'b1': self.b1.copy(),
                    'W2': self.W2.copy(),
                    'b2': self.b2.copy()
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Learning rate decay
            if epoch % 100 == 0 and epoch > 0:
                self.lr *= 0.95
        
        # Restore best weights
        if self.best_weights:
            self.W1 = self.best_weights['W1']
            self.b1 = self.best_weights['b1']
            self.W2 = self.best_weights['W2']
            self.b2 = self.best_weights['b2']
        
        self.training_time = time.time() - start_time
    
    # Keep original fit method for backward compatibility
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        self.fit_with_callback(X, y, X_val, y_val)

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        return a1 @ self.W2 + self.b2


# ======================================================
# ENHANCED DATA LOADER FOR FOOTWEAR SALES
# ======================================================
def load_footwear_data(csv_path, target="units_sold"):
    """
    Load and preprocess footwear sales data for units sold prediction
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Debug: Show column names
        print(f"Columns in dataset: {df.columns.tolist()}")
        print(f"Dataset shape: {df.shape}")
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Create a copy for categorical values extraction
        original_df = df.copy()
        
        # Check if target exists
        if target not in df.columns:
            print(f"Warning: Target column '{target}' not found.")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Try to find alternative target
            target_options = ["units_sold", "quantity", "sales_volume", "total_units", "qty", 
                            "units", "sold_units", "volume", "sales_qty"]
            for alt_target in target_options:
                if alt_target in df.columns:
                    target = alt_target
                    print(f"Using alternative target: {target}")
                    break
        
        # If still not found, check numeric columns
        if target not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # Use the last numeric column as target
                target = numeric_cols[-1]
                print(f"Using last numeric column as target: {target}")
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[target])
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing target values")
        
        # Date Features (only if column exists)
        date_cols = ["order_date", "date", "sale_date", "transaction_date", "purchase_date"]
        date_col_found = None
        for col in date_cols:
            if col in df.columns:
                date_col_found = col
                break
        
        if date_col_found:
            try:
                df["order_date"] = pd.to_datetime(df[date_col_found], errors="coerce")
                df["month"] = df["order_date"].dt.month.fillna(6)
                df["quarter"] = df["order_date"].dt.quarter.fillna(2)
                df["day_of_week"] = df["order_date"].dt.dayofweek.fillna(0)
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
                df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)
                df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
            except Exception as e:
                print(f"Date processing error: {e}")
                df["month"] = 6
                df["quarter"] = 2
                df["day_of_week"] = 0
                df["is_weekend"] = 0
                df["is_month_end"] = 0
                df["is_month_start"] = 0
        else:
            df["month"] = 6
            df["quarter"] = 2
            df["day_of_week"] = 0
            df["is_weekend"] = 0
            df["is_month_end"] = 0
            df["is_month_start"] = 0
        
        # Identify categorical columns (non-numeric with fewer than 50 unique values)
        categorical_cols = []
        for col in df.columns:
            if col != target and df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count < 50 and unique_count > 1:
                    categorical_cols.append(col)
                    print(f"Categorical column: {col} ({unique_count} unique values)")
        
        # If no categorical columns found, use common ones
        if not categorical_cols:
            common_categorical = ["brand", "category", "gender", "sales_channel", 
                                "country", "customer_income_level", "product_type",
                                "color", "size_category"]
            categorical_cols = [col for col in common_categorical if col in df.columns]
        
        # Handle missing values in categorical columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # One-Hot Encoding for categorical columns
        if categorical_cols:
            print(f"One-hot encoding {len(categorical_cols)} categorical columns")
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        else:
            df_encoded = df.copy()
        
        # Identify numerical features (exclude target and date columns)
        numerical_cols = []
        for col in df_encoded.columns:
            if col != target and col not in ['order_date', 'date', 'sale_date', 
                                           'transaction_date', 'purchase_date']:
                if pd.api.types.is_numeric_dtype(df_encoded[col]):
                    numerical_cols.append(col)
        
        print(f"Found {len(numerical_cols)} numerical columns")
        
        # Handle missing values in numerical columns
        for col in numerical_cols:
            if col in df_encoded.columns:
                if df_encoded[col].isna().any():
                    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                    print(f"Filled missing values in {col} with median")
        
        # Feature engineering
        feature_engineering_done = False
        
        # Price and discount features
        price_cols = ["base_price", "price", "unit_price", "cost", "selling_price"]
        discount_cols = ["discount", "discount_percent", "discount_rate", "discount_pct"]
        
        price_col = None
        discount_col = None
        
        for col in price_cols:
            if col in df_encoded.columns:
                price_col = col
                break
        
        for col in discount_cols:
            if col in df_encoded.columns:
                discount_col = col
                break
        
        if price_col and discount_col:
            df_encoded["final_price"] = df_encoded[price_col] * (1 - df_encoded[discount_col] / 100)
            df_encoded["price_discount_ratio"] = df_encoded[price_col] / (df_encoded[discount_col] + 1)
            numerical_cols.extend(["final_price", "price_discount_ratio"])
            feature_engineering_done = True
            print("Created price-related features")
        
        # Rating features
        rating_cols = ["rating", "customer_rating", "review_rating", "product_rating"]
        for col in rating_cols:
            if col in df_encoded.columns:
                df_encoded[f"{col}_bin"] = pd.cut(df_encoded[col], bins=5, labels=False)
                numerical_cols.append(f"{col}_bin")
                feature_engineering_done = True
                print(f"Created binned feature for {col}")
                break
        
        # Size features
        if "size" in df_encoded.columns:
            df_encoded["size_category"] = pd.cut(df_encoded["size"], bins=5, labels=False)
            numerical_cols.append("size_category")
            feature_engineering_done = True
            print("Created size category feature")
        
        # Ensure target column exists
        if target not in df_encoded.columns:
            print(f"Error: Target column '{target}' not found after preprocessing.")
            print(f"Available columns after preprocessing: {df_encoded.columns.tolist()}")
            return None
        
        # Select features and target
        feature_cols = numerical_cols
        
        # Filter to columns that actually exist
        feature_cols = [col for col in feature_cols if col in df_encoded.columns and col != target]
        
        print(f"Using {len(feature_cols)} features for training")
        
        # Check if we have enough data
        if len(df_encoded) < 10:
            print(f"Warning: Very small dataset ({len(df_encoded)} samples)")
        
        if len(feature_cols) == 0:
            print("Error: No features available for training")
            return None
        
        X = df_encoded[feature_cols].values
        y = df_encoded[target].values.reshape(-1, 1)
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Remove any infinite or NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Scale target for better training (especially for neural networks)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_cols,
            "df": df,
            "original_df": original_df,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "categorical_cols": categorical_cols,
            "target_column": target,
            "feature_engineering": feature_engineering_done,
            "dataset_info": {
                "total_samples": len(df),
                "features_count": len(feature_cols),
                "categorical_count": len(categorical_cols),
                "target_mean": float(y.mean()),
                "target_std": float(y.std())
            }
        }
        
    except Exception as e:
        print(f"Error in load_footwear_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ======================================================
# ENHANCED REGRESSION METRICS
# ======================================================
def regression_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Ensure arrays are not empty
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "MAE": 0.0,
            "MSE": 0.0,
            "RMSE": 0.0,
            "R2": 0.0,
            "MAPE": 0.0,
            "Explained Variance": 0.0,
            "Max Error": 0.0,
            "Median Absolute Error": 0.0
        }
    
    # Basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    # Mean Absolute Percentage Error (handle zero values)
    epsilon = 1e-8
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))
        ape = np.where(np.isfinite(ape), ape, 0)
    mape = np.mean(ape) * 100
    
    # Explained variance
    if np.var(y_true) == 0:
        explained_variance = 0.0
    else:
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    # Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_abs_error = np.median(np.abs(y_true - y_pred))
    
    # Directional accuracy
    if len(y_true) > 1:
        dir_acc = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100
    else:
        dir_acc = 0.0
    
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "Explained Variance": float(explained_variance),
        "Max Error": float(max_error),
        "Median Absolute Error": float(median_abs_error),
        "Directional Accuracy": float(dir_acc)
    }


# ======================================================
# ENHANCED PREDICTION HELPER
# ======================================================
def prepare_prediction_input(user_input, feature_names, categorical_cols, original_df, scaler_X):
    """
    Prepare user input for prediction
    user_input: dict with keys matching categorical_cols + numerical features
    """
    try:
        # Start with the original dataframe structure
        df_template = original_df.iloc[[0]].copy()
        
        # Set all values to default (0 for numeric, first value for categorical)
        for col in df_template.columns:
            if pd.api.types.is_numeric_dtype(df_template[col]):
                df_template[col] = 0
            else:
                if col in df_template.columns and len(df_template[col]) > 0:
                    df_template[col] = df_template[col].iloc[0]
                else:
                    df_template[col] = "unknown"
        
        # Set user values
        for key, value in user_input.items():
            # Handle different column name formats
            col_variations = [key, key.lower(), key.replace(' ', '_'), key.replace('_', ' ')]
            
            for col_var in col_variations:
                if col_var in df_template.columns:
                    df_template[col_var] = value
                    break
        
        # Apply the same preprocessing as training
        df_template.columns = df_template.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Date features (if order_date was not provided, use current date)
        if "order_date" not in df_template.columns or pd.isna(df_template["order_date"].iloc[0]):
            df_template["order_date"] = pd.Timestamp.now()
        
        try:
            df_template["order_date"] = pd.to_datetime(df_template["order_date"], errors="coerce")
            df_template["month"] = df_template["order_date"].dt.month.fillna(6)
            df_template["quarter"] = df_template["order_date"].dt.quarter.fillna(2)
            df_template["day_of_week"] = df_template["order_date"].dt.dayofweek.fillna(0)
            df_template["is_weekend"] = df_template["day_of_week"].isin([5, 6]).astype(int)
            df_template["is_month_end"] = df_template["order_date"].dt.is_month_end.astype(int)
            df_template["is_month_start"] = df_template["order_date"].dt.is_month_start.astype(int)
        except:
            df_template["month"] = 6
            df_template["quarter"] = 2
            df_template["day_of_week"] = 0
            df_template["is_weekend"] = 0
            df_template["is_month_end"] = 0
            df_template["is_month_start"] = 0
        
        # Ensure categorical columns exist
        for col in categorical_cols:
            if col not in df_template.columns:
                df_template[col] = "unknown"
        
        # One-hot encode
        df_encoded = pd.get_dummies(df_template, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Feature engineering (matching training)
        # Price and discount features
        price_cols = ["base_price", "price", "unit_price", "cost", "selling_price"]
        discount_cols = ["discount", "discount_percent", "discount_rate", "discount_pct"]
        
        price_col = None
        discount_col = None
        
        for col in price_cols:
            if col in df_encoded.columns:
                price_col = col
                break
        
        for col in discount_cols:
            if col in df_encoded.columns:
                discount_col = col
                break
        
        if price_col and discount_col:
            df_encoded["final_price"] = df_encoded[price_col] * (1 - df_encoded[discount_col] / 100)
            df_encoded["price_discount_ratio"] = df_encoded[price_col] / (df_encoded[discount_col] + 1)
        
        # Rating binning
        rating_cols = ["rating", "customer_rating", "review_rating", "product_rating"]
        for col in rating_cols:
            if col in df_encoded.columns:
                rating_val = df_encoded[col].iloc[0]
                if rating_val < 2: bin_val = 0
                elif rating_val < 3: bin_val = 1
                elif rating_val < 4: bin_val = 2
                elif rating_val < 5: bin_val = 3
                else: bin_val = 4
                df_encoded[f"{col}_bin"] = bin_val
                break
        
        # Size category
        if "size" in df_encoded.columns:
            size_val = df_encoded["size"].iloc[0]
            if size_val < 7: cat = 0
            elif size_val < 9: cat = 1
            elif size_val < 11: cat = 2
            elif size_val < 13: cat = 3
            else: cat = 4
            df_encoded["size_category"] = cat
        
        # Ensure all feature columns exist
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Select and order features exactly as training
        X_pred = df_encoded[feature_names].values
        
        # Scale features
        X_pred_scaled = scaler_X.transform(X_pred)
        
        return X_pred_scaled
        
    except Exception as e:
        print(f"Error in prepare_prediction_input: {str(e)}")
        # Return zeros as fallback
        return np.zeros((1, len(feature_names)))