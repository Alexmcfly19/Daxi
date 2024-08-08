import sqlite3
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import atexit

INITIAL_PRICE = 8000  # Initial fixed cost for every trip
OPTIMAL_TEMPERATURE = 18  # Optimal temperature in degrees Celsius


# Database setup
def setup_database():
    conn = sqlite3.connect('navigation_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS navigation_data (
            id INTEGER PRIMARY KEY,
            distance REAL,
            travel_time REAL,
            temperature REAL,
            weather_condition TEXT,
            area_request TEXT,
            price REAL
        )
    ''')
    conn.commit()
    conn.close()


setup_database()


# Function to insert data
def insert_data(distance, travel_time, temperature, weather_condition, area_request, price):
    conn = sqlite3.connect('navigation_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO navigation_data (distance, travel_time, temperature, weather_condition, area_request, price)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (distance, travel_time, temperature, weather_condition, area_request, price))
    conn.commit()
    conn.close()


# Fetch data from the database
def fetch_data():
    conn = sqlite3.connect('navigation_data.db')
    query = 'SELECT * FROM navigation_data'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# FastAPI app setup
app = FastAPI()


# Pydantic model for input data validation
class Data(BaseModel):
    distance: float
    travel_time: float
    temperature: float
    weather_condition: str
    area_request: str
    price: float = None  # Optional for prediction


# Simple formula for initial predictions
def formula_predict(distance, travel_time, temperature, weather_condition, area_request):
    weather_adjustment = 2000 if weather_condition == 'rainy' else 0
    area_adjustment = 5000 if area_request == 'high_demand' else -1000 if area_request == 'low_demand' else 0
    temperature_penalty = abs(temperature - OPTIMAL_TEMPERATURE) * 1000
    price = INITIAL_PRICE + (distance * 500) + (
                travel_time * 200) + temperature_penalty + weather_adjustment + area_adjustment
    return price


# Train model
model = None
trained_on_data_count = 0


def train_model():
    global trained_on_data_count
    data = fetch_data()
    trained_on_data_count = len(data)
    if trained_on_data_count <= 100:
        return None

    # Preprocess data
    data = pd.get_dummies(data, columns=['weather_condition', 'area_request'])
    data['temperature_penalty'] = abs(data['temperature'] - OPTIMAL_TEMPERATURE) * 1000

    # Extract features and target variable
    features = data[['distance', 'travel_time', 'temperature_penalty',
                     'weather_condition_clear', 'weather_condition_rainy',
                     'area_request_high_demand', 'area_request_low_demand']]
    target = data['price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

    return model


model = train_model()


@app.post('/predict')
async def predict(data: Data):
    distance = data.distance
    travel_time = data.travel_time
    temperature = data.temperature
    weather_condition = data.weather_condition
    area_request = data.area_request

    if trained_on_data_count <= 100:
        # Use formula for predictions
        prediction = formula_predict(distance, travel_time, temperature, weather_condition, area_request)
    else:
        # Feature engineering
        weather_clear = 1 if weather_condition == 'clear' else 0
        weather_rainy = 1 if weather_condition == 'rainy' else 0
        area_high_demand = 1 if area_request == 'high_demand' else 0
        area_low_demand = 1 if area_request == 'low_demand' else 0
        temperature_penalty = abs(temperature - OPTIMAL_TEMPERATURE) * 1000

        features = [[distance, travel_time, temperature_penalty, weather_clear, weather_rainy, area_high_demand,
                     area_low_demand]]

        if model:
            prediction = model.predict(features)[0]
            prediction += INITIAL_PRICE  # Add the initial price to the prediction
        else:
            raise HTTPException(status_code=400, detail="Model not trained. Add data and retrain the model.")

    return {'predicted_price': prediction}


@app.post('/insert')
async def insert(data: Data):
    insert_data(data.distance, data.travel_time, data.temperature, data.weather_condition, data.area_request,
                data.price)
    global model
    model = train_model()
    return {'status': 'Data inserted and model retrained'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)

    # Ensure the database connection is closed on exit
    atexit.register(lambda: sqlite3.connect('navigation_data.db').close())
