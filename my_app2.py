import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 모델들
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

st.title("📈 회귀 분석 통합 GUI 앱 v2")

uploaded_file = st.file_uploader("📤 엑셀 파일 업로드", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ 데이터 업로드 완료!")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_col = st.selectbox("🎯 예측할 Y 변수 선택", columns)
    feature_cols = st.multiselect("🧩 입력 변수 (X) 선택", [c for c in columns if c != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # Train/Test 분할
        test_size = st.slider("Test 데이터 비율 (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # K-Fold 설정
        use_kfold = st.checkbox("K-Fold 교차검증 사용", value=False)
        k = st.slider("K 값 설정 (교차검증)", 2, 10, 5) if use_kfold else None

        # 모델 선택
        model_name = st.selectbox("모델 선택", [
            "Linear Regression", "Polynomial Regression",
            "Ridge", "Lasso", "ElasticNet", "Bayesian Ridge",
            "Decision Tree", "Random Forest", "Gradient Boosting",
            "SVM", "KNN", "Neural Network (MLP)", "Gaussian Process"])

        # 모델 생성 함수
        def get_model(name):
            if name == "Linear Regression":
                return LinearRegression()
            elif name == "Polynomial Regression":
                degree = st.slider("다항식 차수", 2, 5, 2)
                return Pipeline([("poly", PolynomialFeatures(degree=degree)),
                                 ("lin", LinearRegression())])
            elif name == "Ridge":
                alpha = st.number_input("alpha (Ridge)", 0.01, 10.0, 1.0)
                return Ridge(alpha=alpha)
            elif name == "Lasso":
                alpha = st.number_input("alpha (Lasso)", 0.01, 10.0, 1.0)
                return Lasso(alpha=alpha)
            elif name == "ElasticNet":
                alpha = st.number_input("alpha (EN)", 0.01, 10.0, 1.0)
                l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5)
                return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            elif name == "Bayesian Ridge":
                return BayesianRidge()
            elif name == "Decision Tree":
                return DecisionTreeRegressor()
            elif name == "Random Forest":
                return RandomForestRegressor()
            elif name == "Gradient Boosting":
                return GradientBoostingRegressor()
            elif name == "SVM":
                return SVR()
            elif name == "KNN":
                return KNeighborsRegressor()
            elif name == "Neural Network (MLP)":
                return MLPRegressor(max_iter=1000)
            elif name == "Gaussian Process":
                return GaussianProcessRegressor()

        model = get_model(model_name)

        if use_kfold:
            scores = cross_val_score(model, X, y, cv=KFold(n_splits=k, shuffle=True, random_state=42), scoring='r2')
            st.info(f"📊 K-Fold 평균 R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)

            st.subheader("📈 성능 평가 결과")
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")
            st.write(f"**MAE:** {mae:.4f}")

            st.subheader("🧪 예측 결과 시각화")
            chart_data = pd.DataFrame({"실제값": y_test, "예측값": y_pred})
            st.line_chart(chart_data.reset_index(drop=True))

            st.download_button("📥 예측결과 다운로드", chart_data.to_csv(index=False).encode("utf-8-sig"),
                               file_name="예측결과.csv", mime="text/csv")
