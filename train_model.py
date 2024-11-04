import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# Cargar los datos
file_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/wine_quality_df.csv'
data = pd.read_csv(file_path)

# Seleccionar las mejores características manualmente
selected_features = ['volatile_acidity', 'chlorides', 'free_sulfur_dioxide', 'density', 'alcohol']

# Dividir características y objetivo
X = data[selected_features]
y = data['quality']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el pipeline sin SelectKBest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Guardar el modelo y las características seleccionadas
model_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/wine_model.pkl'
features_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/selected_features.pkl'

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

with open(features_path, "wb") as f:
    pickle.dump(selected_features, f)

print("Modelo y características seleccionadas guardados exitosamente.")
