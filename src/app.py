import streamlit as st
from joblib import load
import pandas as pd


@st.cache_data
def get_model(file_name):
    return load(file_name)


def predict(car_features, model):
    return model.predict(car_features)[0]


def create_car_features_df(
    levy,
    manufacturer,
    production_year,
    category,
    leather_interior,
    fuel_type,
    engine_volume,
    mileage,
    cylinders,
    gear_box_type,
    drive_wheels,
    doors,
    wheel,
    color,
    airbags,
    turbo,
):
    has_levy = levy > 0
    car_model = "any"
    age = 2024 - production_year
    mileage_kmh = mileage / age
    return pd.DataFrame(
        [[
            levy, manufacturer, car_model, production_year, category, leather_interior, fuel_type, engine_volume,
            mileage, cylinders, gear_box_type, drive_wheels, doors, wheel, color, airbags, turbo, has_levy, age,
            mileage_kmh,
        ]],
        columns=[
            'Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Engine volume',
            'Mileage', 'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color', 'Airbags', 'Turbo',
            'HasLevy', 'Age', 'Mileage km/h'
        ])


st.title("Car price prediction")

levy_in = st.number_input('Levy', 0)
manufacturer_in = st.selectbox('Manufacturer', (
    'MERCEDES-BENZ', 'JEEP', 'HYUNDAI', 'SSANGYONG', 'NISSAN', 'TOYOTA', 'HONDA', 'DAEWOO', 'KIA', 'SUBARU', 'VAZ',
    'CHEVROLET', 'FORD', 'LEXUS', 'VOLKSWAGEN', 'AUDI', 'MITSUBISHI', 'BMW', 'SEAT', 'DODGE', 'RENAULT', 'FIAT',
    'MAZDA', 'PEUGEOT', 'ACURA', 'SUZUKI', 'OPEL', 'BUICK', 'MINI', 'DAIHATSU', 'SCION', 'JAGUAR', 'CHRYSLER',
    'ALFA ROMEO', 'SKODA', 'LAND ROVER', 'LINCOLN', 'SAAB', 'UAZ', 'GMC', 'MERCURY', 'HAVAL', 'CITROEN', 'PORSCHE',
    'GONOW', 'VOLVO', 'SATURN', 'LANCIA', 'CADILLAC', 'GAZ', 'INFINITI',
))
production_year_in = st.number_input('Prod. year', 2014)
category_in = st.selectbox('Category', (
    'Sedan', 'Jeep', 'Hatchback', 'Minivan', 'Universal', 'Pickup', 'Goods wagon', 'Coupe', 'Microbus', 'Cabriolet'
))
leather_interior_in = st.selectbox("Leather interior", ("Yes", "No"))
fuel_type_in = st.selectbox("Fuel type", ('Petrol', 'LPG', 'Diesel', 'Hybrid', 'Plug-in Hybrid', 'CNG'))
engine_volume_in = st.number_input("Engine volume", 0.8)
mileage_in = st.number_input("Mileage", 0)
cylinders_in = st.number_input("Cylinders", 0)
gear_box_type_in = st.selectbox("Gear box type", ['Tiptronic', 'Automatic', 'Manual', 'Variator'])
drive_wheels_in = st.selectbox("Drive wheels", ['Front', '4x4', 'Rear'])
doors_in = st.selectbox("Doors", ['04-May', '02-Mar', '>5'])
wheel_in = st.selectbox("Wheel", ['Left wheel', 'Right-hand drive'])
color_in = st.selectbox("Color", [
    'Black', 'Silver', 'White', 'Grey', 'Blue', 'Red', 'Green', 'Golden', 'Sky blue', 'Beige', 'Yellow', 'Orange',
    'Brown', 'Carnelian red', 'Pink', 'Purple',
])
airbags_in = st.number_input("Airbags", 0)
turbo_in = st.checkbox("Turbo", False)

if st.button("Estimate price", type="primary"):
    car_features_ = create_car_features_df(
        levy_in,
        manufacturer_in,
        production_year_in,
        category_in,
        leather_interior_in,
        fuel_type_in,
        engine_volume_in,
        mileage_in,
        cylinders_in,
        gear_box_type_in,
        drive_wheels_in,
        doors_in,
        wheel_in,
        color_in,
        airbags_in,
        turbo_in,
    )
    st.write(f"Price: {predict(car_features_, get_model('car_price_prediction.joblib'))}")
