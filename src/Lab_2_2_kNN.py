# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """

        # Validación de los tamaños de X_train y y_train
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")

        # Validación de k y p 
        if (not isinstance(k, int) or k <= 0) or (not isinstance(p, int) or p <= 0):
            raise ValueError("k and p must be positive integers.")
        
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predicciones = []

        # Para cada punto de prueba en X
        for x_test in X:

            # Calcular las distancias desde el punto actual a todos los puntos de entrenamiento
            distancias = self.compute_distances(x_test)
            
            # Obtener los índices de los k vecinos más cercanos
            k_nearest_indices = self.get_k_nearest_neighbors(distancias)
            
            # Obtener las etiquetas de los k vecinos más cercanos
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Determinar la etiqueta más común entre los vecinos más cercanos
            prediccion = self.most_common_label(k_nearest_labels)
            
            # Añadir la predicción a la lista de predicciones
            predicciones.append(prediccion)

        return np.array(predicciones)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """

        # Obtenemos las clases únicas en los datos de entrenamiento
        unique_classes = np.unique(self.y_train)

        # Inicializamos una lista para almacenar las probabilidades predichas de las clases
        proba_predictions = []

        # Iteramos sobre cada muestra en el conjunto de datos de entrada
        for point in X:

            # Calcular las distancias desde el punto actual a todos los puntos de entrenamiento
            distancias = self.compute_distances(point)

            # Obtener los índices de los k vecinos más cercanos
            k_nearest_indices = self.get_k_nearest_neighbors(distancias)

            # Obtener las etiquetas de los k vecinos más cercanos
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Inicializamos un diccionario para contar las ocurrencias de cada clase
            class_counts = {}
            for clase in unique_classes:
                class_counts[clase] = 0

            # Contamos las ocurrencias de cada clase en los k vecinos más cercanos
            for label in k_nearest_labels:
                class_counts[label] += 1

            # Calculamos las probabilidades dividiendo las ocurrencias entre k
            probabilidades = []
            for clase in unique_classes:
                probabilidad = class_counts[clase] / self.k
                probabilidades.append(probabilidad)

            # Añadir las probabilidades a la lista de predicciones
            proba_predictions.append(probabilidades)

        return np.array(proba_predictions)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        
        distancias = []  # Lista donde vamos guardando cada distancia

        # Para cada punto en el conjunto de entrenamiento
        for x_train in self.x_train:

            # Calculamos la distancia de minkowski desde point hasta x_train
            distancia = minkowski_distance(point, x_train, self.p)

            # Añadimos la distancia a la lista
            distancias.append(distancia)

        return np.array(distancias)


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        
        if distances.size == 0:
            raise ValueError("La matriz de distancias está vacía")
        
        # Creamos un vector con las distancias ordenadas (de la más corta a la más larga)
        sorted_distances = np.argsort(distances)

        # Cogemos los vecinos más cercano (utilizamos la k que indica el número de vecinos que utilizamos)
        k_nearest_neighbors = sorted_distances[:self.k]

        return k_nearest_neighbors

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        # Verificar que knn_labels no esté vacío
        if knn_labels.size == 0:
            raise ValueError("El arreglo de etiquetas de los vecinos está vacío.")
        
        # Obtener los valores únicos y sus frecuencias
        unique_labels, counts = np.unique(knn_labels, return_counts=True)
        
        # Encontrar el índice del valor con la mayor frecuencia
        most_common_index = np.argmax(counts)
        
        # Obtener la etiqueta más común
        most_common_label = unique_labels[most_common_index]

        return most_common_label

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"


def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    # Verdaderos Positivos (TP): Predice positivo y es positivo
    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    # Verdaderos Negativos (TN): Predice negativo y es negativo
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    # Falsos Positivos (FP): Predice positivo y es negativo
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    # Falsos Negativos (FN): Predice negativo y es positivo
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    # Mapeamos las etiquetas verdaderas a valores binarios (0 y 1)
    y_true_binary = []
    for label in y_true:
        if label == positive_label:
            y_true_binary.append(1)
        else:
            y_true_binary.append(0)
    y_true_binary = np.array(y_true_binary)

    # Inicializamos las listas para almacenar las fracciones de positivos y las probabilidades promedio
    true_proportions = []
    predicted_probabilities = []

    # Calculamos el tamaño de cada bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Iteramos sobre cada bin
    for i in range(n_bins):
        # Determinamos los índices de las muestras que caen dentro del bin actual
        bin_mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i + 1])
        bin_samples = y_probs[bin_mask]
        bin_labels = y_true_binary[bin_mask]

        # Si el bin contiene muestras, calculamos la fracción de positivos y la probabilidad promedio
        if len(bin_samples) > 0:
            true_proportion = np.mean(bin_labels)
            predicted_probability = np.mean(bin_samples)
        else:
            # Si el bin está vacío, asignamos NaN para evitar problemas en el gráfico
            true_proportion = np.nan
            predicted_probability = np.nan

        # Añadimos los resultados a las listas
        true_proportions.append(true_proportion)
        predicted_probabilities.append(predicted_probability)

    # Convertimos las listas a arrays de NumPy
    true_proportions = np.array(true_proportions)
    predicted_probabilities = np.array(predicted_probabilities)

    # Creamos el gráfico de la curva de calibración
    plt.figure(figsize=(8, 6))
    plt.plot(predicted_probabilities, true_proportions, marker='o', linestyle='-', label='Curva de Calibración')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibración Perfecta')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Fracción de Positivos')
    plt.title('Curva de Calibración')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    # Mapeamos las etiquetas verdaderas a valores binarios (0 y 1)
    y_true_mapped = []
    for label in y_true:
        if label == positive_label:
            y_true_mapped.append(1)
        else:
            y_true_mapped.append(0)
    y_true_mapped = np.array(y_true_mapped)

    # Separamos las probabilidades según las etiquetas
    positive_probs = y_probs[y_true_mapped == 1]
    negative_probs = y_probs[y_true_mapped == 0]

    # Dibujamos los histogramas
    plt.figure(figsize=(10, 6))
    plt.hist(positive_probs, bins=n_bins, alpha=0.5, label='Clase Positiva', color='blue')
    plt.hist(negative_probs, bins=n_bins, alpha=0.5, label='Clase Negativa', color='red')
    plt.xlabel('Probabilidad Predicha')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades para las Clases Positiva y Negativa')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }


def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """

    # Mapeamos las etiquetas a valores binarios (positivo/negativo)
    y_true_mapped = []
    for label in y_true:
        if label == positive_label:
            y_true_mapped.append(1)
        else:
            y_true_mapped.append(0)
    y_true_mapped = np.array(y_true_mapped)

    # Ordenamos las probabilidades y las etiquetas verdaderas de mayor a menor probabilidad
    thresholds = np.linspace(0, 1, 11)

    # Inicializamos las listas para las tasas FPR y TPR
    tpr = []
    fpr = []

    # Calculamos FPR y TPR para cada umbral
    for threshold in thresholds:
        # Predicciones binarizadas basadas en el umbral
        y_pred = (y_probs >= threshold).astype(int)

        # Calculamos la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR)
        tp = np.sum((y_pred == 1) & (y_true_mapped == 1))  # Verdaderos positivos
        fp = np.sum((y_pred == 1) & (y_true_mapped == 0))  # Falsos positivos
        fn = np.sum((y_pred == 0) & (y_true_mapped == 1))  # Falsos negativos
        tn = np.sum((y_pred == 0) & (y_true_mapped == 0))  # Verdaderos negativos

        # TPR (Tasa de Verdaderos Positivos) = TP / (TP + FN)
        # FPR (Tasa de Falsos Positivos) = FP / (FP + TN)
        tpr.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) != 0 else 0)

    # Convertimos a arrays para compatibilidad con matplotlib
    tpr = np.array(tpr)
    fpr = np.array(fpr)

    # Dibujamos de la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='Curva ROC')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}
