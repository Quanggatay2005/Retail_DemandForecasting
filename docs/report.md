# kSynerX Demand Forecasting Project Report

## Problem Statement

Developing a robust demand forecasting system for fresh retail presents several unique challenges that this project aims to solve:

1.  **Censored demand data**: Historical sales data does not always reflect true customer demand. The system must account for this "censorship" to avoid under-forecasting future demand based on historical stockouts.
2.  **High demand volatility & complex external factors**: The demand for fresh produce, meat, and seafood is extremely sensitive to various factors. For example, demand volatility can be observed in the sharp intra-day fluctuations of product sales. Furthermore, consumer behavior is heavily influenced by promotional campaigns, price changes, and even weather conditions. For instance, heavy rain often discourages in-store shopping, leading to a surge in online delivery orders.
3.  **Automated inventory integration**: The retail chain's inventory management system requires a seamless, automated gateway to communicate with the forecasting models. Instead of manual data exports, the system needs an API endpoint where it can programmatically query the model and receive immediate predictions.
4.  **Security and system maintainability**: As the system handles sensitive business data and integrates directly with core inventory operations, ensuring robust API security is vital. 

## Solutions Implemented

To address the challenges outlined above, the project implements the following key solutions:

1.  **Robust Data Processing Pipeline (`data_preparation.py`)**: The system features a comprehensive data pipeline that directly downloads raw records from the `Dingdong-Inc/FreshRetailNet-50K` dataset via Hugging Face. The data is parsed, flattened, and rigorously cleaned. A critical part of this pipeline is the intelligent interpolation of missing weather data using linear methods combined with forward and backward filling, ensuring continuous time-series validity. Outliers in sales data are handled via winsorization rather than deletion to preserve holiday and promotional signals.
2.  **Advanced Time-Series Modeling (`mlforecast_train.py`)**: The project trains and deploys advanced ML models (Pipeline with Ridge Regression, XGBoost, and LightGBM) utilizing the `mlforecast` framework. This framework automatically and safely generates lag features and rolling window statistics without risking data leakage, completely superseding older manual feature engineering logic.
3.  **Scalable Model Serving with Ray**: To fulfill the need for automated inventory integration, the project implements a robust REST API endpoint using `FastAPI` wrapped dynamically within **Ray Serve**. External systems can seamlessly transmit store information and product codes, and immediately receive the calculated forecast predictions. Ray Serve ensures this API is highly scalable, maintainable, and capable of handling concurrent requests efficiently.
4.  **Business-Centric Model Evaluation**: We use robust cross-validation methods (time-series splitting) native to `mlforecast`. The models are evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to determine the best-performing algorithm.

## System Architecture

The kSynerX Demand Forecasting project is designed as an end-to-end Machine Learning pipeline for time-series forecasting. The architecture is modular and consists of four main components:

1.  **Data Ingestion & Cleaning (`src/data_preparation.py`)**: 
    - Fetches the raw `Dingdong-Inc/FreshRetailNet-50K` dataset from the Hugging Face Hub.
    - Standardizes column names, parses datetime fields, and flattens arrays into daily aggregations.
    - Handles missing weather data seamlessly using linear interpolation mapped over locations.
    - Applies winsorization to cap outliers (preventing extreme noise while preserving real promotional spikes).
2.  **Automated Feature Engineering & Model Training (`src/mlforecast_train.py`)**:
    - Central training pipeline completely automating time-series structure via the `mlforecast` library.
    - Autogenerates temporal features, lag features, and expanding/rolling window features without data leakage.
    - Dynamically trains multiple algorithms concurrently: `Pipeline(SimpleImputer, Ridge)`, `XGBoost`, and `LightGBM`.
    - Integrates Time-Series Cross-Validation and evaluates the models based on predictive metrics (MAE, RMSE, MAPE).
3.  **Model Serving (`src/serve.py` & `src/ray_serve.py`)**:
    - **`serve.py`**: Contains the core FastAPI application logic, including the health check routes and endpoint definitions for forecasting inference.
    - **`ray_serve.py`**: Uses **Ray Serve** integrated with the FastAPI app (`@serve.ingress`) to orchestrate deployment, handle multi-replica concurrency, and manage dynamic model loading in a production-ready REST API.

---

## 2. Implemented Features & Features Not Yet Implemented

### Implemented Features
- **Hugging Face Integration**: Successfully transitioned to fetching real-world retail datasets (`FreshRetailNet-50K`) directly from HF Hub.
- **Data Interpolation**: Sophisticated weather data handling using linear interpolation and forward/backward filling.
- **Automated ML Pipeline**: Robust generation of lags and rolling windows using `mlforecast`, completely removing the need for manual feature engineering scripts.
- **Multi-Model Support**: Support for Ridge Regression (with missing value imputation), XGBoost, and LightGBM models correctly handling dynamic exogenous variables.
- **Location-Specific Filtering**: The pipeline dynamically filters data based on geographical locations (city ID) to optimize RAM usage.
- **Scalable Serving API**: Deployed via Ray Serve and FastAPI, seamlessly routing predict requests and maintaining health checks.

### Features Not Yet Implemented (Future Work)
- **Advanced Hyperparameter Tuning**: Currently, models use default or hardcoded hyperparameters. Tools like Optuna could be integrated.
- **Real-Time Data Streaming**: Integration with Kafka for real-time feature streaming and prediction logging is pending.
- **Automated CI/CD & Retraining**: No automated pipeline yet to retrain models periodically as new data arrives.
- **BI Dashboard**: A graphical interface (e.g., Streamlit) to visualize forecasts vs. actuals is not yet connected to this specific repository.

---

## 3. Human vs. AI Contributions (Mandatory Breakdown)

This project was developed using a pair-programming approach between the human developer and AI coding tools.

*   **Parts Written by the User (Human)**:
    - **System Design & Architecture**: Decided on the overall pipeline flow (Data Preparation -> MLForecast Trainer -> FastAPI Server -> Ray Serve Wrapper).
    - **Technology Stack Selection**: Chose FastAPI, Ray Serve, LightGBM/XGBoost, and `mlforecast`.
    - **Code Cleanups & Pipeline Deprecation**: Identified and orchestrated the removal of legacy systems (the manual feature engineering scripts and the old `train.py`) to streamline architecture.
    - **Dataset Selection**: Identified and integrated the Dingdong-Inc/FreshRetailNet-50K dataset via the Hugging Face ecosystem.
    - **Model Evaluation**: Decided to use MAE, RMSE, and WAPE as evaluation metrics.
    

*   **Parts Implemented using AI Tools (Gemini/Copilot)**:
    - **Advanced Data Interpolation**: The AI authored the linear interpolation and filling methodologies in the data preparation sequence.
    - **MLForecast Bug Fixes**: The AI identified and corrected severe `ValueError` exceptions caused by MLForecast misclassifying dynamic exogenous features as static. The AI also repaired the prediction pipelines, providing the correct aligned shapes and indices.
    - **Ridge Imputation Logic**: The AI wrapped Ridge estimators in robust `SimpleImputer` Pipelines to prevent models from crashing on NaN values generated naturally by trailing lag calculations.
    - **Report Generation**: This very documentation file was drafted by AI based on the project's source code and continually updated to reflect the evolving structure.

---

## 4. Lessons Learned

1.  **Robust data filtering and schema validation**: When switching from dummy data to real-world data, unexpected data types (like string representations of arrays) can easily break machine learning pipelines. So, robust data filtering and schema validation are critical early steps.
2. **Implemetation of MLForecast**: MLForecast is a powerful library for time-series forecasting, but it requires careful handling of exogenous variables. 
3. **Basics of MLOps**: implentation of data pipeline, ML model training, model serving, and model evaluation is the basics of MLOps.
4. **Basics of FastAPI and Ray Serve**: FastAPI is a web framework for building APIs, and Ray Serve is a framework for deploying and scaling distributed applications. 

## 5. Some Experiments & Extra Analysis
- Try to implement a pipeline with FastAPI only (not including mlforecast and ray serve)
- AI assisted: 
    + I use NotebookLM to help me understand the concept in this project, including datasets in HuggingFace, the article included. 
    + Then I use Claude to create 3 pipeline and structures to make the project as it is today.
    + For implemenation, I use Gemini Chat and Antigravity to help with coding. 
- Choose between 2 project structures: single file vs multiple files. Decided to use multiple files to make it easier to manage and scale.
- 
