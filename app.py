import streamlit as st
import pandas as pd
import pickle

# ---------------- TARGET ENCODING CLASS ----------------
class TargetEncoding:
    def __init__(self, alpha=10):
        self.alpha = alpha
        self.global_mean = None
        self.mappings = {}

    def fit(self, df, cat_cols, target_col):
        self.global_mean = df[target_col].mean()
        for col in cat_cols:
            stats = df.groupby(col)[target_col].agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            smooth = (counts * means + self.alpha * self.global_mean) / (counts + self.alpha)
            self.mappings[col] = smooth

    def transform(self, df, cat_cols):
        df = df.copy()
        for col in cat_cols:
            df[col + '_tencoded'] = df[col].map(self.mappings[col])
            df[col + '_tencoded'] = df[col + '_tencoded'].fillna(self.global_mean)
        df = df.drop(columns=cat_cols)
        return df

    def fit_transform(self, df, cat_cols, target_col):
        self.fit(df, cat_cols, target_col)
        return self.transform(df, cat_cols)

# ---------------- LOAD CSV OPTIONS ----------------
def load_csv_options(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        options = df.iloc[:, 0].dropna().astype(str).tolist()
        return [opt for opt in options if opt != ""]
    except Exception as e:
        st.error(f"❌ Failed to load {file_path}: {e}")
        return []

# Load options
districts = sorted(load_csv_options("csvs/location_district_name.csv"))
mahallas = sorted(load_csv_options("csvs/mahalla.csv"))
location_main_options = load_csv_options("csvs/location_main.csv")
house_type_options = load_csv_options("csvs/house_type.csv")
gas_options = load_csv_options("csvs/gas.csv")
electricity_options = load_csv_options("csvs/electricity.csv")
# heating_options = load_csv_options("csvs/heating.csv") #TODO: cant read csv smhw

# Heating types — from your input
heating_options = [
    "no_heating", "mixed", "gas", "central",
    "electric", "solid_fuel", "liquid_fuel"
]

# Placeholders
water_options = ["central", "well", "none"]
wc_house_options = ["inside", "outside", "shared", "none"]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_target_encoder():
    try:
        with open("target_encoder_v2.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load target_encoder_v2.pkl: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        with open("lgbm_model_v3.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load lgbm_model_v3.pkl: {e}")
        return None

model = load_model()
target_encoder = load_target_encoder()

if model is None or target_encoder is None:
    st.stop()

# ---------------- UI ----------------
st.set_page_config(page_title="🏡 House Price Predictor", page_icon="💰")
st.title("🏡 House Price Prediction (Uzbekistan)")
st.markdown("Enter property details for accurate price estimation.")

# Input columns
col1, col2 = st.columns(2)

with col1:
    business = st.selectbox("Business-related?", [0, 1], index=0)
    price_negotiable = st.selectbox("Price negotiable?", [0, 1], index=1)
    number_of_rooms = st.number_input("Number of rooms", 1, 20, value=3)
    total_area = st.number_input("Total area (m²)", 1, 50000, value=60)
    total_living_area = st.number_input("Living area (m²)", 1, 10000, value=40)
    total_floors = st.number_input("Total floors", 1, 30, value=5)
    ceiling_height = st.number_input("Ceiling height (cm)", 200, 400, value=280)
    furnished_house = st.selectbox("Furnished?", [0, 1], index=0)

with col2:
    year_of_construction_sale = st.number_input("Year built", 1900, 2025, value=2000)
    comission = st.selectbox("Commission?", [0, 1], index=0)
    location_main = st.selectbox(
        "Location (main)",
        location_main_options,
        index=location_main_options.index("В городе") if "В городе" in location_main_options else 0
    )
    house_type = st.selectbox(
        "House type",
        house_type_options,
        index=house_type_options.index("brick") if "brick" in house_type_options else 0
    )
    heating = st.selectbox(
        "Heating system",
        heating_options,
        index=heating_options.index("central") if "central" in heating_options else 0
    )
    gas = st.selectbox(
        "Gas supply",
        gas_options,
        index=gas_options.index("central") if "central" in gas_options else 0
    )
    electricity = st.selectbox(
        "Electricity",
        electricity_options,
        index=electricity_options.index("connected") if "connected" in electricity_options else 0
    )
    water = st.selectbox("Water supply", water_options, index=0)
    wc_house = st.selectbox("Toilet location", wc_house_options, index=0)

# Location
st.subheader("📍 Location")
location_district_name = st.selectbox("District", districts)
mahalla = st.selectbox("Mahalla", mahallas)

# Multilabel features
st.subheader("🔧 House Features")

house_repairs = st.multiselect(
    "Repairs",
    [
        "Авторский проект", "Евроремонт", "Не достроен", "Под снос",
        "Предчистовая отделка", "Средний ремонт", "Требует ремонта",
        "Черновая отделка"
    ],
    default=["Средний ремонт"]
)

more_house = st.multiselect(
    "Extras",
    [
        "Air_conditioner", "Basement", "Bathhouse", "Garage", "Garden", "Gym",
        "Home_appliances", "Internet", "Satellite_TV", "Sauna", "Security",
        "Sewerage", "Storage_room", "Swimming_pool", "Telephone"
    ],
    default=["Internet", "Home_appliances"]
)

near_is = st.multiselect(
    "Nearby",
    [
        "Bus_stops", "Cafe", "Entertainment_venues", "Green_area", "Hospital",
        "Kindergarten", "Park", "Parking_lot", "Playground", "Polyclinic",
        "Restaurants", "School", "Shops", "Supermarket"
    ],
    default=["School", "Shops", "Bus_stops"]
)

# ---------------- HELPER: MULTILABEL WITH UNDERSCORE FIX ----------------
def encode_multilabel(df, column_name, classes_list):
    for cls in classes_list:
        safe_cls = cls.replace(" ", "_")  #  fixes space → underscore
        df[f"{column_name}_{safe_cls}"] = df[column_name].apply(lambda x: 1 if cls in x else 0)
    df.drop(columns=[column_name], inplace=True)
    return df

# ---------------- BUILD INPUT DF ----------------
def build_input_df():
    df = pd.DataFrame([{
        "business": business,
        "price_negotiable": price_negotiable,
        "number_of_rooms": number_of_rooms,
        "total_area": total_area,
        "total_living_area": total_living_area,
        "total_floors": total_floors,
        "ceiling_height": ceiling_height,
        "furnished_house": furnished_house,
        "year_of_construction_sale": year_of_construction_sale,
        "comission": comission,
        "house_repairs": house_repairs,
        "more_house": more_house,
        "near_is": near_is,
        "location_district_name": location_district_name,
        "mahalla": mahalla,
        "location_main": location_main,
        "house_type": house_type,
        "heating": heating,
        "gas": gas,
        "electricity": electricity,
        "water": water,
        "wc_house": wc_house,


        "private_house_type": house_type  # ← safe copy
    }])

    # Encode multilabels
    df["house_repairs"] = df["house_repairs"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "house_repairs", [
        "Авторский проект", "Евроремонт", "Не достроен", "Под снос",
        "Предчистовая отделка", "Средний ремонт", "Требует ремонта",
        "Черновая отделка"
    ])

    df["more_house"] = df["more_house"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "more_house", [
        "Air_conditioner", "Basement", "Bathhouse", "Garage", "Garden", "Gym",
        "Home_appliances", "Internet", "Satellite_TV", "Sauna", "Security",
        "Sewerage", "Storage_room", "Swimming_pool", "Telephone"
    ])

    df["near_is"] = df["near_is"].apply(lambda x: x if isinstance(x, list) else [])
    df = encode_multilabel(df, "near_is", [
        "Bus_stops", "Cafe", "Entertainment_venues", "Green_area", "Hospital",
        "Kindergarten", "Park", "Parking_lot", "Playground", "Polyclinic",
        "Restaurants", "School", "Shops", "Supermarket"
    ])

    # Encode all categorical columns
    cat_cols = [
        "location_district_name", "mahalla", "location_main",
        "house_type", "private_house_type", "heating", "gas",
        "electricity", "water", "wc_house"
    ]
    # Keep only those in encoder
    available = [col for col in cat_cols if col in target_encoder.mappings]
    encoded = target_encoder.transform(df[available], available)
    df = pd.concat([df.drop(columns=available), encoded], axis=1)

    # All 57 model features exist?
    for feat in model.feature_names_in_:
        if feat not in df.columns:
            df[feat] = 0
            st.warning(f"⚠️ Missing feature '{feat}' → filled with 0")

    df = df[model.feature_names_in_]
    return df

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Price", type="primary"):
    try:
        df = build_input_df()
        pred = model.predict(df)[0]

        st.success(f"💰 **Predicted Price**: ${pred:,.0f}")
        st.info(f"✅ Input has **{len(df.columns)}** features (expected: 57)")

        # for debugging purps
        df.to_csv("sample.csv", index=False)
        st.download_button(
            "📥 Download sample.csv",
            data=open("sample.csv", "rb").read(),
            file_name="sample.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.code(str(e))