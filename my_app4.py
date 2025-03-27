import streamlit as st
import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

st.set_page_config(layout="wide")
st.title("📈 회귀 분석 통합 GUI 앱 v4 (Optuna 설정 + 예측률 히트맵 포함)")

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

        test_size = st.slider("Test 데이터 비율 (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        use_kfold = st.checkbox("K-Fold 교차검증 사용", value=False)
        k = st.slider("K 값 설정 (교차검증)", 2, 10, 5) if use_kfold else None

        use_optuna = st.checkbox("Optuna 자동 하이퍼파라미터 튜닝 사용", value=False)
        n_trials = st.slider("Optuna 반복 횟수 (n_trials)", 10, 100, 30) if use_optuna else None

        model_name = st.selectbox("모델 선택", [
            "Linear Regression", "Polynomial Regression",
            "Ridge", "Lasso", "ElasticNet", "Bayesian Ridge",
            "Decision Tree", "Random Forest", "Gradient Boosting",
            "SVM", "KNN", "Neural Network (MLP)", "Gaussian Process"])

        def get_model(name, trial=None):
            if name == "Linear Regression":
                return LinearRegression()
            elif name == "Polynomial Regression":
                degree = trial.suggest_int("degree", 2, 5) if trial else st.slider("다항식 차수", 2, 5, 2)
                return Pipeline([("poly", PolynomialFeatures(degree=degree)),
                                 ("lin", LinearRegression())])
            elif name == "Ridge":
                alpha = trial.suggest_float("alpha", 0.01, 10.0) if trial else st.number_input("alpha (Ridge)", 0.01, 10.0, 1.0)
                return Ridge(alpha=alpha)
            elif name == "Lasso":
                alpha = trial.suggest_float("alpha", 0.01, 10.0) if trial else st.number_input("alpha (Lasso)", 0.01, 10.0, 1.0)
                return Lasso(alpha=alpha)
            elif name == "ElasticNet":
                alpha = trial.suggest_float("alpha", 0.01, 10.0) if trial else st.number_input("alpha (EN)", 0.01, 10.0, 1.0)
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if trial else st.slider("l1_ratio", 0.0, 1.0, 0.5)
                return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            elif name == "Bayesian Ridge":
                return BayesianRidge()
            elif name == "Decision Tree":
                return DecisionTreeRegressor()
            elif name == "Random Forest":
                n_estimators = trial.suggest_int("n_estimators", 50, 200) if trial else 100
                max_depth = trial.suggest_int("max_depth", 3, 15) if trial else None
                return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
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

        if use_optuna:
            st.info("🔄 Optuna로 자동 튜닝 중... 시간이 걸릴 수 있습니다")

            def objective(trial):
                model = get_model(model_name, trial)
                score = cross_val_score(model, X, y, cv=3, scoring='r2')
                return np.mean(score)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            st.success(f"✅ Optuna 최적 R²: {study.best_value:.4f}")
            model = get_model(model_name, trial=study.best_trial)

        else:
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

            def calc_accuracy(y_true, y_pred):
                return np.where(y_true != 0,
                                (1 - np.abs(y_true - y_pred) / np.abs(y_true)) * 100,
                                np.nan)

            accuracy = calc_accuracy(y_test.values, y_pred)

            st.subheader("📈 성능 평가 결과")
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**MSE:** {mse:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")
            st.write(f"**MAE:** {mae:.4f}")
            st.write(f"**평균 예측률:** {np.nanmean(accuracy):.2f}%")

            st.subheader("🧪 예측 결과 시각화")
            result_df = pd.DataFrame(X_test.reset_index(drop=True))
            result_df['실제값'] = y_test.values
            result_df['예측값'] = y_pred
            result_df['예측률 (%)'] = accuracy

            st.line_chart(result_df[['실제값', '예측값']].reset_index(drop=True))

            st.subheader("🎯 예측률 히트맵")
            fig, ax = plt.subplots(figsize=(6, len(result_df) * 0.25))
            sns.heatmap(result_df[['예측률 (%)']], cmap='RdYlGn', annot=True, fmt=".1f", ax=ax)
            st.pyplot(fig)

            st.download_button("📥 예측결과 다운로드", result_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="예측결과.csv", mime="text/csv")
