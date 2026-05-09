import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a;
    }
    [data-testid="stHeader"] {
        background-color: #1a1a1a;
    }
    .main {
        padding-top: 0rem;
        background-color: #1a1a1a;
    }
    .stMetric {
        background-color: #44ab41;
        padding: 10px;
        border-radius: 10px;
    }
    [data-testid="metric-container"] {
        background-color: #90EE90;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #228B22;
    }
    [data-testid="metric-container"] label {
        color: #000 !important;
    }
    [data-testid="metric-container"] div {
        color: #000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    """Train the model using the Housing.csv dataset"""
    # Load data
    dataset = pd.read_csv('Housing.csv')
    
    # Store original for reference
    original_dataset = dataset.copy()
    
    # Label encode categorical features
    label_encoders = {}
    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']
    
    for col in categorical_cols:
        le = LabelEncoder()
        dataset[f'{col}_code'] = le.fit_transform(dataset[col])
        label_encoders[col] = le
    
    # Remove outliers
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if not c.endswith("_code")]
    
    Q1 = dataset[numeric_cols].quantile(0.25)
    Q3 = dataset[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outlier_mask = (dataset[numeric_cols] < (Q1 - 1.5 * IQR)) | (dataset[numeric_cols] > (Q3 + 1.5 * IQR))
    dataset = dataset.loc[~outlier_mask.any(axis=1)].reset_index(drop=True)
    
    # Scale area
    area_scaler = StandardScaler()
    dataset['area_scaled'] = area_scaler.fit_transform(dataset[['area']])
    
    # Feature list
    feature_list = [
        'area_scaled', 'bedrooms', 'bathrooms', 'stories', 'parking',
        'mainroad_code', 'guestroom_code', 'basement_code',
        'hotwaterheating_code', 'airconditioning_code',
        'prefarea_code', 'furnishingstatus_code'
    ]
    
    # Prepare features and target
    X = dataset[feature_list]
    y = dataset['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, area_scaler, label_encoders, feature_list, importance_df

# Load model
model, area_scaler, label_encoders, feature_list, importance_df = train_model()

# Title and description
st.title("🏠 House Price Predictor")
st.markdown("Predict house prices based on features using Machine Learning (Random Forest)")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Enter House Details")
    
    # Create form
    with st.form("prediction_form"):
        # Numeric inputs
        col_a, col_b = st.columns(2)
        
        with col_a:
            area = st.number_input(
                "Area (sq ft)",
                min_value=1000,
                max_value=15000,
                value=6000,
                step=100
            )
            bedrooms = st.slider(
                "Bedrooms",
                min_value=1,
                max_value=10,
                value=4
            )
            bathrooms = st.slider(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=2
            )
            stories = st.slider(
                "Stories",
                min_value=1,
                max_value=5,
                value=2
            )
        
        with col_b:
            parking = st.slider(
                "Parking Spaces",
                min_value=0,
                max_value=5,
                value=1
            )
            mainroad = st.selectbox(
                "On Main Road?",
                options=["Yes", "No"],
                index=0
            )
            guestroom = st.selectbox(
                "Guest Room?",
                options=["Yes", "No"],
                index=1
            )
            basement = st.selectbox(
                "Basement?",
                options=["Yes", "No"],
                index=0
            )
        
        # Boolean features
        col_c, col_d = st.columns(2)
        
        with col_c:
            hotwaterheating = st.selectbox(
                "Hot Water Heating?",
                options=["Yes", "No"],
                index=1
            )
            airconditioning = st.selectbox(
                "Air Conditioning?",
                options=["Yes", "No"],
                index=1
            )
        
        with col_d:
            prefarea = st.selectbox(
                "Preferred Area?",
                options=["Yes", "No"],
                index=1
            )
            furnishingstatus = st.selectbox(
                "Furnishing Status",
                options=["furnished", "semi-furnished", "unfurnished"],
                index=0
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

with col2:
    st.subheader("📊 Feature Importance")
    fig = px.bar(
        importance_df.head(8),
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Make prediction
if submitted:
    try:
        # Create raw house data
        raw_new_house = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad': mainroad.lower(),
            'guestroom': guestroom.lower(),
            'basement': basement.lower(),
            'hotwaterheating': hotwaterheating.lower(),
            'airconditioning': airconditioning.lower(),
            'prefarea': prefarea.lower(),
            'furnishingstatus': furnishingstatus
        }
        
        # Prepare features
        new_house_df = pd.DataFrame([raw_new_house])
        
        # Scale area
        new_house_df['area_scaled'] = area_scaler.transform(new_house_df[['area']])
        
        # Encode categorical features
        for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']:
            new_house_df[f'{col}_code'] = label_encoders[col].transform(new_house_df[[col]])
        
        # Select features in correct order
        new_house_df = new_house_df[feature_list]
        
        # Make prediction
        predicted_price = model.predict(new_house_df)[0]
        
        # Display result
        st.divider()
        
        col_result1, col_result2 = st.columns([1, 1])
        
        with col_result1:
            st.metric(
                label="💰 Predicted House Price",
                value=f"Rs {predicted_price:,.2f}",
                delta=None
            )
        
        with col_result2:
            # Price range based on prediction
            lower_range = predicted_price * 0.85
            upper_range = predicted_price * 1.15
            st.info(f"📈 Estimated Range: Rs {lower_range:,.0f} - Rs {upper_range:,.0f}")
        
        # Show input summary
        st.subheader("📝 Input Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.write(f"**Area:** {area:,} sq ft")
            st.write(f"**Bedrooms:** {bedrooms}")
            st.write(f"**Bathrooms:** {bathrooms}")
            st.write(f"**Stories:** {stories}")
        
        with summary_col2:
            st.write(f"**Parking:** {parking}")
            st.write(f"**Main Road:** {mainroad}")
            st.write(f"**Guest Room:** {guestroom}")
            st.write(f"**Basement:** {basement}")
        
        with summary_col3:
            st.write(f"**Hot Water Heating:** {hotwaterheating}")
            st.write(f"**Air Conditioning:** {airconditioning}")
            st.write(f"**Preferred Area:** {prefarea}")
            st.write(f"**Furnishing:** {furnishingstatus}")
    
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")

# Footer with info
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; margin-top: 2rem;'>
<p><small>🤖 Powered by Random Forest Machine Learning Model</small></p>
<p><small>Dataset: Housing.csv | Model Accuracy: High Performance</small></p>
</div>
""", unsafe_allow_html=True)
