import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from ollama import chat


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TrafficFlow AI",
    page_icon="🚦",
    layout="wide"
)

# =========================================================
# YELLOW THEME
# =========================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #FFF9E6;
    }

    /* Main content area */
    .main {
        background-color: #FFF9E6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFD54F;
    }

    [data-testid="stSidebar"] * {
        color: #3E2723;
    }

    /* Main headings */
    h1 {
        color: #8A5A00;
        font-weight: 700;
    }

    h2 {
        color: #9A6700;
        font-weight: 650;
    }

    h3 {
        color: #A66A00;
    }

    /* Normal text */
    p {
        color: #4E4030;
    }

    /* Buttons */
    .stButton > button {
        background-color: #FBC02D;
        color: #3E2723;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #F9A825;
        color: white;
    }

    /* Input boxes */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {
        background-color: #FFFDF5;
        border: 1px solid #E6B800;
        border-radius: 8px;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFDF5;
        border: 1px solid #E6B800;
        border-radius: 8px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #FFF3B0;
        border: 1px solid #F4C430;
        padding: 15px;
        border-radius: 12px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #E6B800;
        border-radius: 10px;
    }

    /* Horizontal divider */
    hr {
        border-color: #E6B800;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv("traffic_volume.csv")

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

    data["year"] = data["date_time"].dt.year
    data["month"] = data["date_time"].dt.month
    data["day"] = data["date_time"].dt.day
    data["hour"] = data["date_time"].dt.hour

    data["day_of_week"] = (
        data["date_time"].dt.dayofweek
    )

    data["is_weekend"] = data["day_of_week"].apply(
        lambda x: 1 if x >= 5 else 0
    )

    return data


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "traffic_model.pkl"
    )

    feature_columns = joblib.load(
        "model_features.pkl"
    )

    return model, feature_columns


# =========================================================
# GEMMA 4 FUNCTION
# =========================================================

def get_gemma_explanation(
    prediction,
    year,
    month,
    day,
    hour,
    day_of_week,
    temperature,
    rainfall,
    snowfall,
    clouds,
    weather_main,
    weather_description
):

    prompt = f"""
You are an AI traffic analysis assistant.

A Machine Learning Random Forest model predicted:

{prediction:.0f} vehicles

The input conditions were:

Year: {year}
Month: {month}
Day: {day}
Hour: {hour}
Day of week: {day_of_week}

Temperature: {temperature:.2f}
Rainfall in last 1 hour: {rainfall:.2f} mm
Snowfall in last 1 hour: {snowfall:.2f} mm
Cloud coverage: {clouds}%

Weather: {weather_main}
Weather description: {weather_description}

Your task is to explain the machine learning prediction.

Give the response in very simple language.

Include exactly these sections:

1. Predicted Traffic
2. Traffic Level
3. Possible Reasons
4. Short Advice

Traffic level should be described as:
- LOW
- MODERATE
- HIGH

Use ONLY the information provided above.

Do not invent traffic accidents, road closures,
events, accidents, or other information.

Keep the answer concise and suitable for a student ML project.
"""

    try:

        response = chat(
            model="gemma4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful traffic analysis "
                        "assistant for a Machine Learning project."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content

    except Exception as e:

        return (
            "⚠️ Gemma 4 is currently unavailable.\n\n"
            "Please make sure Ollama is running and "
            "the Gemma 4 model is installed."
        )


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    df = load_dataset()

    model, feature_columns = load_model()

except Exception as e:

    st.error(
        "⚠️ Unable to load the required project files."
    )

    st.info(
        """
Make sure these files are present:

• traffic_volume.csv
• traffic_model.pkl
• model_features.pkl
"""
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚦 TrafficFlow AI")

st.sidebar.write(
    "Traffic Volume Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Choose Page",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Analytics",
        "🤖 Model Information",
        "💬 Gemma 4 Assistant"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
🚗 Traffic Volume Prediction

Machine Learning + Generative AI

Built with:

• Python
• Pandas
• Scikit-learn
• Random Forest
• Plotly
• Streamlit
• Ollama
• Gemma 4
"""
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.title("🚦 TrafficFlow AI")

    st.subheader(
        "Intelligent Traffic Volume Prediction using Machine Learning"
    )

    st.write(
        """
This application predicts traffic volume using
weather conditions and time-related information.

Gemma 4 is used as an AI assistant to explain
the machine learning prediction in simple language.
"""
    )

    st.divider()

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    average_traffic = df["traffic_volume"].mean()
    maximum_traffic = df["traffic_volume"].max()
    minimum_traffic = df["traffic_volume"].min()
    average_temperature = df["temp"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🚗 Average Traffic",
            f"{average_traffic:,.0f}"
        )

    with col2:

        st.metric(
            "🚦 Maximum Traffic",
            f"{maximum_traffic:,.0f}"
        )

    with col3:

        st.metric(
            "📉 Minimum Traffic",
            f"{minimum_traffic:,.0f}"
        )

    with col4:

        st.metric(
            "🌡️ Average Temperature",
            f"{average_temperature:.1f}"
        )

    st.divider()

    # -----------------------------------------------------
    # TRAFFIC BY HOUR
    # -----------------------------------------------------

    st.header("📈 Traffic Volume by Hour")

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .reset_index()
    )

    fig_hour = px.line(
        hourly_traffic,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TEMPERATURE + CLOUDS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌡️ Temperature vs Traffic")

        fig_temp = px.scatter(
            df,
            x="temp",
            y="traffic_volume",
            title="Temperature vs Traffic",
            opacity=0.65
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True
        )

    with col2:

        st.subheader("☁️ Cloud Coverage vs Traffic")

        fig_cloud = px.scatter(
            df,
            x="clouds_all",
            y="traffic_volume",
            title="Cloud Coverage vs Traffic",
            opacity=0.65
        )

        st.plotly_chart(
            fig_cloud,
            use_container_width=True
        )

    # -----------------------------------------------------
    # WEATHER GRAPH
    # -----------------------------------------------------

    st.header("🌦️ Average Traffic by Weather")

    weather_traffic = (
        df.groupby("weather_main")["traffic_volume"]
        .mean()
        .reset_index()
    )

    fig_weather = px.bar(
        weather_traffic,
        x="weather_main",
        y="traffic_volume",
        title="Average Traffic by Weather"
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Prediction":

    st.title("🔮 Traffic Prediction")

    st.write(
        "Enter weather and time information to predict traffic volume."
    )

    st.divider()

    # -----------------------------------------------------
    # TIME INFORMATION
    # -----------------------------------------------------

    st.header("📅 Time Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        year = st.number_input(
            "Year",
            min_value=2018,
            max_value=2035,
            value=2025
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )

    with col2:

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=1
        )

        hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=8
        )

    with col3:

        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_of_week = st.selectbox(
            "Day of Week",
            days
        )

        day_number = days.index(
            day_of_week
        )

        is_weekend = (
            1 if day_number >= 5 else 0
        )

    st.divider()

    # -----------------------------------------------------
    # WEATHER INFORMATION
    # -----------------------------------------------------

    st.header("🌦️ Weather Information")

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "🌡️ Temperature",
            value=float(df["temp"].mean())
        )

        rainfall = st.number_input(
            "🌧️ Rainfall (last 1 hour)",
            min_value=0.0,
            value=0.0
        )

        snowfall = st.number_input(
            "❄️ Snowfall (last 1 hour)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        clouds = st.slider(
            "☁️ Cloud Coverage (%)",
            min_value=0,
            max_value=100,
            value=50
        )

        weather_main_values = sorted(
            df["weather_main"]
            .dropna()
            .unique()
            .tolist()
        )

        weather_description_values = sorted(
            df["weather_description"]
            .dropna()
            .unique()
            .tolist()
        )

        weather_main = st.selectbox(
            "🌦️ Weather",
            weather_main_values
        )

        weather_description = st.selectbox(
            "📝 Weather Description",
            weather_description_values
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚦 Predict Traffic Volume",
        use_container_width=True
    ):

        # -------------------------------------------------
        # CREATE INPUT DATA
        # -------------------------------------------------

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        # -------------------------------------------------
        # NUMERICAL FEATURES
        # -------------------------------------------------

        if "temp" in input_data.columns:
            input_data["temp"] = temperature

        if "rain_1h" in input_data.columns:
            input_data["rain_1h"] = rainfall

        if "snow_1h" in input_data.columns:
            input_data["snow_1h"] = snowfall

        if "clouds_all" in input_data.columns:
            input_data["clouds_all"] = clouds

        if "year" in input_data.columns:
            input_data["year"] = year

        if "month" in input_data.columns:
            input_data["month"] = month

        if "day" in input_data.columns:
            input_data["day"] = day

        if "hour" in input_data.columns:
            input_data["hour"] = hour

        if "day_of_week" in input_data.columns:
            input_data["day_of_week"] = day_number

        if "is_weekend" in input_data.columns:
            input_data["is_weekend"] = is_weekend

        # -------------------------------------------------
        # WEATHER MAIN
        # -------------------------------------------------

        weather_main_column = (
            "weather_main_" + weather_main
        )

        if weather_main_column in input_data.columns:

            input_data[
                weather_main_column
            ] = 1

        # -------------------------------------------------
        # WEATHER DESCRIPTION
        # -------------------------------------------------

        weather_description_column = (
            "weather_description_"
            + weather_description
        )

        if weather_description_column in input_data.columns:

            input_data[
                weather_description_column
            ] = 1

        # -------------------------------------------------
        # RANDOM FOREST PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]

        prediction = max(
            0,
            prediction
        )

        # -------------------------------------------------
        # TRAFFIC LEVEL
        # -------------------------------------------------

        average_traffic = df[
            "traffic_volume"
        ].mean()

        if prediction < average_traffic * 0.75:

            traffic_level = "LOW"

        elif prediction < average_traffic * 1.25:

            traffic_level = "MODERATE"

        else:

            traffic_level = "HIGH"

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🚦 Prediction Result"
        )

        st.success(
            f"Predicted Traffic Volume: "
            f"{prediction:,.0f} vehicles"
        )

        if traffic_level == "LOW":

            st.success(
                "🟢 Traffic Level: LOW"
            )

        elif traffic_level == "MODERATE":

            st.warning(
                "🟡 Traffic Level: MODERATE"
            )

        else:

            st.error(
                "🔴 Traffic Level: HIGH"
            )

        # -------------------------------------------------
        # GEMMA 4 EXPLANATION
        # -------------------------------------------------

        st.subheader(
            "🤖 Gemma 4 AI Explanation"
        )

        with st.spinner(
            "Gemma 4 is analyzing the prediction..."
        ):

            gemma_explanation = get_gemma_explanation(
                prediction=prediction,
                year=year,
                month=month,
                day=day,
                hour=hour,
                day_of_week=day_of_week,
                temperature=temperature,
                rainfall=rainfall,
                snowfall=snowfall,
                clouds=clouds,
                weather_main=weather_main,
                weather_description=weather_description
            )

        st.info(
            gemma_explanation
        )

        # -------------------------------------------------
        # PREDICTION DETAILS
        # -------------------------------------------------

        st.header(
            "📋 Prediction Details"
        )

        result_data = pd.DataFrame({

            "Parameter": [
                "Year",
                "Month",
                "Day",
                "Hour",
                "Day of Week",
                "Temperature",
                "Rainfall",
                "Snowfall",
                "Cloud Coverage",
                "Weather",
                "Weather Description",
                "Predicted Traffic"
            ],

            "Value": [
                year,
                month,
                day,
                hour,
                day_of_week,
                temperature,
                rainfall,
                snowfall,
                clouds,
                weather_main,
                weather_description,
                f"{prediction:,.0f} vehicles"
            ]

        })

        st.dataframe(
            result_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.title("📊 Traffic Analytics")

    st.write(
        "Explore traffic patterns and relationships in the dataset."
    )

    st.divider()

    # -----------------------------------------------------
    # TRAFFIC BY HOUR
    # -----------------------------------------------------

    st.header(
        "🕐 Average Traffic Volume by Hour"
    )

    hourly_data = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .reset_index()
    )

    fig1 = px.line(
        hourly_data,
        x="hour",
        y="traffic_volume",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TEMPERATURE + CLOUDS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🌡️ Temperature vs Traffic"
        )

        fig2 = px.scatter(
            df,
            x="temp",
            y="traffic_volume",
            title="Temperature vs Traffic",
            opacity=0.65
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "☁️ Cloud Coverage vs Traffic"
        )

        fig3 = px.scatter(
            df,
            x="clouds_all",
            y="traffic_volume",
            title="Cloud Coverage vs Traffic",
            opacity=0.65
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # -----------------------------------------------------
    # WEATHER GRAPH
    # -----------------------------------------------------

    st.header(
        "🌦️ Average Traffic by Weather"
    )

    weather_data = (
        df.groupby("weather_main")["traffic_volume"]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        weather_data,
        x="weather_main",
        y="traffic_volume",
        title="Average Traffic by Weather"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.header(
        "🌲 Random Forest Feature Importance"
    )

    try:

        feature_importance = pd.DataFrame({

            "Feature": feature_columns,

            "Importance":
            model.feature_importances_

        })

        feature_importance = (
            feature_importance
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        top_features = (
            feature_importance
            .head(10)
        )

        fig5 = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

        st.subheader(
            "📋 Feature Importance Table"
        )

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )

    except Exception:

        st.info(
            "Feature importance is available for tree-based models."
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.title(
        "🤖 Machine Learning Model"
    )

    st.write(
        "Information about the algorithms used in the project."
    )

    st.divider()

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "📈 Linear Regression"
        )

        st.write(
            """
Predicts traffic volume by finding
a linear relationship between input
features and traffic volume.
"""
        )

    with col2:

        st.subheader(
            "🌳 Decision Tree"
        )

        st.write(
            """
Uses decision rules and tree branches
to predict traffic volume.
"""
        )

    with col3:

        st.subheader(
            "🌲 Random Forest"
        )

        st.write(
            """
Combines multiple decision trees
to make the final prediction.
"""
        )

    st.divider()

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.header(
        "📁 Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Rows",
            len(df)
        )

    with col2:

        st.metric(
            "Dataset Columns",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Model Features",
            len(feature_columns)
        )

    st.divider()

    # -----------------------------------------------------
    # MODEL FEATURES
    # -----------------------------------------------------

    st.header(
        "🧩 Model Features"
    )

    feature_table = pd.DataFrame({

        "Feature":
        feature_columns

    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # PROJECT DESCRIPTION
    # -----------------------------------------------------

    st.header(
        "ℹ️ About This Project"
    )

    st.info(
        """
🚦 Traffic Volume Prediction

This Machine Learning project predicts
traffic volume using weather conditions
and time-related features.

Machine Learning:

• Linear Regression
• Decision Tree
• Random Forest

Generative AI:

• Gemma 4

Gemma 4 is used to explain the
Random Forest prediction in simple language.
"""
    )


# =========================================================
# GEMMA 4 ASSISTANT
# =========================================================

elif page == "💬 Gemma 4 Assistant":

    st.title(
        "💬 Gemma 4 Traffic Assistant"
    )

    st.write(
        """
Ask Gemma 4 questions about traffic,
machine learning, or the prediction system.
"""
    )

    st.divider()

    user_question = st.text_area(
        "Ask your question:",
        placeholder=(
            "Example: Why can traffic volume be "
            "higher during some hours?"
        ),
        height=120
    )

    if st.button(
        "🤖 Ask Gemma 4",
        use_container_width=True
    ):

        if user_question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            assistant_prompt = f"""
You are Gemma 4, an AI assistant
inside a Traffic Volume Prediction
Machine Learning project.

The project uses:

- Random Forest
- Traffic volume data
- Temperature
- Rainfall
- Snowfall
- Cloud coverage
- Weather
- Date and time features

User question:

{user_question}

Answer in simple language.

If the question is about the project,
explain it like a beginner-friendly
Machine Learning teacher.

Do not invent information about
the dataset.
"""

            try:

                with st.spinner(
                    "Gemma 4 is thinking..."
                ):

                    response = chat(
                        model="gemma4",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful "
                                    "Machine Learning "
                                    "and traffic analysis "
                                    "assistant."
                                )
                            },
                            {
                                "role": "user",
                                "content": assistant_prompt
                            }
                        ]
                    )

                st.subheader(
                    "🤖 Gemma 4 Response"
                )

                st.write(
                    response.message.content
                )

            except Exception:

                st.error(
                    """
Gemma 4 could not be reached.

Make sure:

1. Ollama is installed.
2. Ollama is running.
3. Gemma 4 is installed.

Run:

ollama pull gemma4

Then:

ollama run gemma4
"""
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚦 TrafficFlow AI | "
    "Machine Learning + Generative AI | "
    "Random Forest • Streamlit • Ollama • Gemma 4"
)