from flask import Flask, render_template, request
import pandas as pd
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense

app = Flask(__name__)
CORS(app)

# Load your dataset


# Replace 'path/to/your/dataset.csv' with the actual path to your dataset file


file_path = 'indiancrop_datasetwith_via.xlsx'


df = pd.read_excel(file_path) 
X = df[['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']]
y = df['CROP']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) 
def rf(test_data):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    print("random forest")

    print("Accuracy: {:.2%}".format(accuracy))
    # print("Classification Report:\n", report)
    df_test = pd.DataFrame(test_data)
    predictions = rf_model.predict(df_test)
    print("Predicted Crop:", predictions)
    return predictions[0]
def nav(test_data):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("naive_bayes")
    print("Accuracy: {:.2%}".format(accuracy))
    # print("Classification Report:\n", report)
    df_test = pd.DataFrame(test_data)
    predictions = nb_model.predict(df_test)

    print("Predicted Crop:", predictions)
    return predictions[0]
def dt(test_data):
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("DecisionTree")
    print("Accuracy: {:.2%}".format(accuracy))
    # print("Classification Report:\n", report)
    df_test = pd.DataFrame(test_data)
    predictions = dt_model.predict(df_test)
    print("Predicted Crop:", predictions)
    return predictions[0]

def knn(test_data, k_neighbors=3):
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("knn")
    print("Accuracy: {:.2%}".format(accuracy))
    # print("Classification Report:\n", report)

    df_test = pd.DataFrame(test_data)
    predictions = knn_model.predict(df_test)

    print("Predicted Crop:", predictions)
    return predictions[0]
def ltsm(N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL):
    # Load the dataset
    data = pd.read_excel('indiancrop_datasetwith_via.xlsx')

    data = data[['N_SOIL', 'P_SOIL', 'K_SOIL',
                'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'CROP']]

    print(data.head())
    # Extract features and labels

    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # Use LabelEncoder to convert string labels to numerical labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training features
    features_scaled = scaler.fit_transform(features)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, test_size=0.2, random_state=42)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=len(np.unique(labels)), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test))

    # Example new data point for prediction with 6 features
    # Replace with your actual values
    new_data_point = np.array(
        [[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL]])
    # new_data_point = np.array([[80, 50, 42, 22.5, 75.0, 6.8, 3]])

    # Normalize the new data using the fitted scaler
    new_data_scaled = scaler.transform(new_data_point)

    # Reshape the new data for LSTM input
    new_data_reshaped = np.reshape(
        new_data_scaled, (1, 1, new_data_scaled.shape[1]))

    # Make prediction
    prediction = model.predict(new_data_reshaped)

    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    print("Predicted Crop:", predicted_label[0])
    return predicted_label[0]


# Ant colony
def antcol(desired_state, desired_crop):
    print("working")
    file_path = 'indiancrop_datasetwith_via.xlsx'
    df = pd.read_excel(file_path)
    selected_columns = ['STATE', 'CROP_PRICE', 'CROP',
                        'DESTINATION', 'DISTANCE', "VIA1", "VIA2", "VIA3"]
    filtered_df = df[selected_columns]
    filtered_df = df.loc[(df['STATE'] == desired_state) & (
        df['CROP'] == desired_crop), selected_columns]
    result_df = filtered_df.groupby(['STATE', 'DESTINATION', "DISTANCE",
                                    "CROP", "VIA1", "VIA2", "VIA3"], as_index=False)['CROP_PRICE'].mean()
    data = result_df
    np_data = np.array([data['CROP_PRICE'], data['DISTANCE']]).T
    n_ants = 5
    n_iterations = 100
    decay = 0.6
    alpha = 1.0
    beta = 2.0
    pheromone = np.ones(len(np_data))
    best_solution_index = -1
    best_solution_distance = float('inf')
    best_solution_price = float('inf')
    best_solution_vias = []
    for i in range(n_iterations):
        ants = np.zeros((n_ants, len(np_data)), dtype=int)
        for ant in range(n_ants):
            for j in range(len(np_data)):
                choices = np.delete(np.arange(len(np_data)), ants[ant, :j])
                probabilities = (pheromone[choices] ** alpha) * \
                    ((1.0 / np_data[choices][:, 1]) ** beta)
                probabilities /= probabilities.sum()
                selected_index = np.random.choice(choices, p=probabilities)
                ants[ant, j] = selected_index
        pheromone *= (1.0 - decay)
        for ant in range(n_ants):
            for j in range(len(np_data) - 1):
                pheromone[ants[ant, j]] += 1.0 / np_data[ants[ant, j], 0]
        min_distance_index = np.argmin(np_data[:, 1])
        current_solution_distance = np_data[min_distance_index, 1]
        current_solution_price = np_data[min_distance_index, 0]
        if current_solution_distance < best_solution_distance:
            best_solution_distance = current_solution_distance
            best_solution_price = current_solution_price
            best_solution_index = min_distance_index
            best_solution_vias = data.loc[best_solution_index, [
                "VIA1", "VIA2", "VIA3"]].values.tolist()
    s={"desired_state":desired_state,
                "designation":data.loc[best_solution_index, "DESTINATION"],
                "Distance":best_solution_distance,
                "price":best_solution_price,
                "Vias":best_solution_vias}
    
    return s
# artifical bee colony
def artibeecol(desired_state, desired_crop):
    s=[]
    file_path = 'indiancrop_datasetwith_via.xlsx'
    df = pd.read_excel(file_path)
    selected_columns = ['STATE', 'CROP_PRICE', 'CROP',
                        'DESTINATION', 'DISTANCE', "VIA1", "VIA2", "VIA3"]
    filtered_df = df[selected_columns]
    # desired_state = 'Gujarat'
    # desired_crop = 'Maize'
    filtered_df = df.loc[(df['STATE'] == desired_state) & (
        df['CROP'] == desired_crop), selected_columns]
    result_df = filtered_df.groupby(['STATE', 'DESTINATION', "DISTANCE",
                                    "CROP", "VIA1", "VIA2", "VIA3"], as_index=False)['CROP_PRICE'].mean()
    data = result_df
    np_data = np.array([data['CROP_PRICE'], data['DISTANCE']]).T
    n_bees = 5
    n_iterations = 100
    limit = 100
    lower_bound = 0
    upper_bound = len(np_data)
    best_solution_index = -1
    best_solution_distance = float('inf')
    best_solution_price = float('inf')
    best_solution_vias = []
    for iteration in range(n_iterations):
        for bee in range(n_bees):
            current_solution_index = np.random.randint(
                lower_bound, upper_bound)
            current_solution_distance = np_data[current_solution_index, 1]
            current_solution_price = np_data[current_solution_index, 0]
            if current_solution_distance < best_solution_distance:
                best_solution_distance = current_solution_distance
                best_solution_price = current_solution_price
                best_solution_index = current_solution_index
                best_solution_vias = data.loc[best_solution_index, [
                    "VIA1", "VIA2", "VIA3"]].values.tolist()
        if np.random.rand() < 1.0 / (1 + best_solution_distance):
            best_solution_index = np.random.randint(lower_bound, upper_bound)
            best_solution_distance = np_data[best_solution_index, 1]
            best_solution_price = np_data[best_solution_index, 0]
            best_solution_vias = data.loc[best_solution_index, [
                "VIA1", "VIA2", "VIA3"]].values.tolist()
        # print(f"Iteration {iteration + 1}/{n_iterations}: Best Distance = {best_solution_distance}, Best Price = {best_solution_price}")
        
    s={"desired_state":desired_state,
                       "designation":data.loc[best_solution_index, "DESTINATION"],
                       "Distance":best_solution_distance,
                       "price":best_solution_price,
                       "Vias":best_solution_vias}
    
    return s
def simulated_annealing(desired_state, desired_crop, initial_temperature=1000, cooling_rate=0.95, n_iterations=100):
    s=[]
    print(desired_state, desired_crop)
    file_path = 'indiancrop_datasetwith_via.xlsx'
    df = pd.read_excel(file_path)
    selected_columns = ['STATE', 'CROP_PRICE', 'CROP',
                        'DESTINATION', 'DISTANCE', "VIA1", "VIA2", "VIA3"]
    filtered_df = df[selected_columns]

    filtered_df = df.loc[(df['STATE'] == desired_state) & (
        df['CROP'] == desired_crop), selected_columns]

    result_df = filtered_df.groupby(['STATE', 'DESTINATION', "DISTANCE",
                                    "CROP", "VIA1", "VIA2", "VIA3"], as_index=False)['CROP_PRICE'].mean()

    data = result_df
    np_data = np.array([data['CROP_PRICE'], data['DISTANCE']]).T

    current_solution_index = np.random.randint(0, len(np_data))
    current_solution_distance = np_data[current_solution_index, 1]
    current_solution_price = np_data[current_solution_index, 0]
    current_solution_vias = data.loc[current_solution_index, [
        "VIA1", "VIA2", "VIA3"]].values.tolist()

    best_solution_index = current_solution_index
    best_solution_distance = current_solution_distance
    best_solution_price = current_solution_price
    best_solution_vias = current_solution_vias

    temperature = initial_temperature

    for iteration in range(n_iterations):
        neighbor_index = np.random.randint(0, len(np_data))
        neighbor_distance = np_data[neighbor_index, 1]
        neighbor_price = np_data[neighbor_index, 0]
        neighbor_vias = data.loc[neighbor_index, [
            "VIA1", "VIA2", "VIA3"]].values.tolist()

        if neighbor_distance < current_solution_distance or np.random.rand() < np.exp((current_solution_distance - neighbor_distance) / temperature):
            current_solution_index = neighbor_index
            current_solution_distance = neighbor_distance
            current_solution_price = neighbor_price
            current_solution_vias = neighbor_vias

        if current_solution_distance < best_solution_distance:
            best_solution_index = current_solution_index
            best_solution_distance = current_solution_distance
            best_solution_price = current_solution_price
            best_solution_vias = current_solution_vias

        temperature *= cooling_rate
    s={"desired_state":desired_state,
                       "designation":data.loc[best_solution_index, "DESTINATION"],
                       "Distance":best_solution_distance,
                       "price":best_solution_price,
                       "Vias":best_solution_vias}

    
    return s


def get_source_states_for_crop(crop):
    file_path = 'indiancrop_datasetwith_via.xlsx'
    df = pd.read_excel(file_path)

    # Filter the data for the specified crop
    filtered_df = df[df['CROP'] == crop]

    # Get unique source states for the filtered crop
    source_states = filtered_df['STATE'].unique()

    return source_states

# Define the route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# Define the route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Parse form data
    N_SOIL = request.form['N_SOIL']
    P_SOIL = request.form['P_SOIL']
    K_SOIL = request.form['K_SOIL']
    TEMPERATURE = request.form['TEMPERATURE']
    HUMIDITY = request.form['HUMIDITY']
    ph = request.form['ph']
    RAINFALL = request.form['RAINFALL']
    algorithm = request.form['algorithm']
    optimization = request.form['optimization']

    # Create test data dictionary
    test_data = {
        'N_SOIL': [int(N_SOIL)],
        'P_SOIL': [int(P_SOIL)],
        'K_SOIL': [int(K_SOIL)],
        'TEMPERATURE': [float(TEMPERATURE)],
        'HUMIDITY': [float(HUMIDITY)],
        'ph': [float(ph)],
        'RAINFALL': [float(RAINFALL)],
    }
    
    # Call the appropriate function based on the selected algorithm
    if algorithm == '1':
        crop = rf(test_data)
    elif algorithm == '2':
        crop = nav(test_data)
    elif algorithm == '3':
        crop = dt(test_data)
    elif algorithm == '4':
        crop = knn(test_data)
    elif algorithm == '5':
        crop = ltsm(test_data["N_SOIL"][0], test_data["P_SOIL"][0], test_data["K_SOIL"][0], test_data["TEMPERATURE"]
                    [0], test_data['HUMIDITY'][0], test_data['ph'][0], test_data['RAINFALL'][0])
    
    

    source_states = get_source_states_for_crop(crop)
    optimization_method = { '1': 'Simulated Annealing', '2': 'Artificial Bee Colony', '3': 'Ant Colony' }
    out=[]
    for source in source_states:
        print(source)
        if optimization == '1':
            data=simulated_annealing(source, crop)
            
        elif optimization == '2':
            data=artibeecol(source, crop)
            
        else :
            data=antcol(source, crop)
        print(data)
        out.append(data)
    
    
    
    return render_template('result.html', crop=crop, optimization_method=optimization_method[optimization],out=out)

if __name__ == '__main__':
    app.run(debug=True)
