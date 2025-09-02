import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys
sys.stdout.reconfigure(encoding="utf-8")

# ==== 0) PODESIVI PARAMETRI ====
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GEN_PATH = DATA_DIR / "Plant_1_Generation_Data.csv"
WEA_PATH = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Ako hoces da filtriras noc (kada je irradiation == 0), podesi na True:
FILTER_NIGHT = False

# ==== 1) UCITAVANJE I SPAJANJE ====
gen = pd.read_csv(GEN_PATH)
weather = pd.read_csv(WEA_PATH)

gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], errors="coerce")
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
gen = gen.dropna(subset=["DATE_TIME"])
weather = weather.dropna(subset=["DATE_TIME"])

# agregacija: ukupna snaga po vremenu (preko svih invertera)
gen_agg = (gen.groupby(["PLANT_ID","DATE_TIME"], as_index=False)
             .agg(AC_POWER=("AC_POWER","sum"), DC_POWER=("DC_POWER","sum")))

# agregacija: proseci senzora po vremenu (preko svih senzora)
weather_agg = (weather.groupby(["PLANT_ID","DATE_TIME"], as_index=False)
               .agg(AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE","mean"),
                    MODULE_TEMPERATURE=("MODULE_TEMPERATURE","mean"),
                    IRRADIATION=("IRRADIATION","mean")))

df = pd.merge(gen_agg, weather_agg, on=["PLANT_ID","DATE_TIME"], how="inner")
df = df.sort_values("DATE_TIME").reset_index(drop=True)

# ==== 2) FEATURE ENGINEERING ====
df["HOUR"] = df["DATE_TIME"].dt.hour
df["DAY"] = df["DATE_TIME"].dt.day
df["MONTH"] = df["DATE_TIME"].dt.month
df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek

# ciklicni
df["HOUR_SIN"] = np.sin(2*np.pi*df["HOUR"]/24)
df["HOUR_COS"] = np.cos(2*np.pi*df["HOUR"]/24)
df["MONTH_SIN"] = np.sin(2*np.pi*df["MONTH"]/12)
df["MONTH_COS"] = np.cos(2*np.pi*df["MONTH"]/12)

# basic clean
df = df.dropna(subset=["AC_POWER","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"]).copy()

if FILTER_NIGHT:
    df = df[df["IRRADIATION"] > 0].copy()

# ==== 3) TRAIN / TEST SPLIT (vremenski svesno) ====
feature_cols = [
    "HOUR","DAY","MONTH","DAY_OF_WEEK",
    "HOUR_SIN","HOUR_COS","MONTH_SIN","MONTH_COS",
    "AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","IRRADIATION"
]

X = df[feature_cols].astype(float)
y = df["AC_POWER"].astype(float)

split_idx = int(len(df)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==== 4) MODELI ====
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

# (Opcionalno) XGBoost ako postoji
try:
    from xgboost import XGBRegressor
    models["XGBRegressor"] = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, tree_method="hist"
    )
except Exception as e:
    print("XGBoost nije instaliran (preskacem).", e)

# ==== 5) TRENIRANJE + METRIKE ====
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

results = []
preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae, rmse, r2 = evaluate(y_test, y_pred)
    preds[name] = y_pred
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:>17} | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")

results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
results_df.to_csv(OUT_DIR/"model_metrics.csv", index=False)

# ==== 6) VIZUALIZACIJE ZA NAJBOLJI MODEL ====
best_name = results_df.iloc[0]["model"]
print("\nNajbolji model:", best_name)
y_pred_best = preds[best_name]

# scatter
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_best, s=10, alpha=0.6)
mn, mx = float(min(y_test.min(), y_pred_best.min())), float(max(y_test.max(), y_pred_best.max()))
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("Stvarna AC_POWER"); plt.ylabel("Predikcija AC_POWER")
plt.title(f"{best_name}: Stvarne vs Predikcije")
plt.tight_layout()
plt.savefig(OUT_DIR/"scatter_actual_vs_pred.png", dpi=150)
plt.close()

# serija (poslednjih 300 tacaka)
N = 300 if len(y_test) > 300 else len(y_test)
plt.figure(figsize=(10,4))
plt.plot(y_test.values[-N:], label="Stvarno")
plt.plot(y_pred_best[-N:], label="Predikcija")
plt.title(f"{best_name}: poslednjih {N} tacaka (test)")
plt.xlabel("Korak"); plt.ylabel("AC_POWER"); plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR/"timeseries_last_300.png", dpi=150)
plt.close()

# feature importance ako model podrzava
best_model = models[best_name]
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8,5))
    plt.barh(np.array(feature_cols)[order], importances[order])
    plt.xlabel("Vaznost"); plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR/"feature_importance.png", dpi=150)
    plt.close()

# ==== 7) EXPORT PREDIKCIJA ====
pd.DataFrame({
    "y_test": y_test.values,
    f"y_pred_{best_name}": y_pred_best
}).to_csv(OUT_DIR/"test_predictions.csv", index=False)


months = sorted(df["MONTH"].dropna().astype(int).unique())
n = len(months)
cols = min(4, n)
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
fig.suptitle("OLAP slojevi — Prosečna AC_POWER po Satu i Danu za svaki Mesec", y=1.02)

hours = np.arange(24)
days = np.arange(1, 32)

# globalne granice skale
vmin = None
vmax = None
tmp_vals = []
for m in months:
    sub = df[df["MONTH"] == m]
    if sub.empty: 
        continue
    mat = np.full((len(hours), len(days)), np.nan)
    grp = sub.groupby(["HOUR","DAY"])["AC_POWER"].mean()
    for (h,d), val in grp.items():
        if (0 <= h <= 23) and (1 <= d <= 31):
            mat[h, d-1] = val
    tmp = mat[~np.isnan(mat)]
    if tmp.size:
        tmp_vals.append(tmp)
if tmp_vals:
    all_vals = np.concatenate(tmp_vals)
    vmin, vmax = float(np.percentile(all_vals, 5)), float(np.percentile(all_vals, 95))

for i, m in enumerate(months):
    r, c = divmod(i, cols)
    ax = axes[r][c]
    sub = df[df["MONTH"] == m]
    mat = np.full((len(hours), len(days)), np.nan)
    grp = sub.groupby(["HOUR","DAY"])["AC_POWER"].mean()
    for (h,d), val in grp.items():
        if (0 <= h <= 23) and (1 <= d <= 31):
            mat[h, d-1] = val
    im = ax.imshow(mat, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"Mesec {m}")
    ax.set_xlabel("Dan"); ax.set_ylabel("Sat")
    ax.set_xticks([0,6,13,20,30]); ax.set_xticklabels([1,7,14,21,31])
    ax.set_yticks([0,6,12,18,23]); ax.set_yticklabels([0,6,12,18,23])

# sakrij prazne axe-ove
for j in range(i+1, rows*cols):
    r, c = divmod(j, cols)
    axes[r][c].axis("off")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
cbar.set_label("Prosečna AC_POWER")
plt.tight_layout()
fig.savefig(OUT_DIR/"olap_layers_by_month.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Dodatni preseci (opciono):
# HOUR x MONTH
mat_hm = np.full((24, int(df["MONTH"].max())), np.nan)
grp_hm = df.groupby(["HOUR","MONTH"])["AC_POWER"].mean()
for (h, m), v in grp_hm.items():
    if 0 <= h <= 23:
        mat_hm[h, int(m)-1] = v
plt.figure(figsize=(8,4))
plt.imshow(mat_hm, aspect="auto", origin="lower")
plt.title("OLAP presek — Prosečna AC_POWER (Sat × Mesec)")
plt.xlabel("Mesec"); plt.ylabel("Sat")
plt.xticks(range(int(df["MONTH"].max())), range(1, int(df["MONTH"].max())+1))
plt.yticks([0,6,12,18,23],[0,6,12,18,23])
cbar = plt.colorbar(); cbar.set_label("Prosečna AC_POWER")
plt.tight_layout()
plt.savefig(OUT_DIR/"olap_hour_by_month.png", dpi=150, bbox_inches="tight")
plt.close()

# HOUR x DAY (ceo skup)
mat_hd = np.full((24, 31), np.nan)
grp_hd = df.groupby(["HOUR","DAY"])["AC_POWER"].mean()
for (h, d), v in grp_hd.items():
    if (0 <= h <= 23) and (1 <= d <= 31):
        mat_hd[h, d-1] = v
plt.figure(figsize=(10,4))
plt.imshow(mat_hd, aspect="auto", origin="lower")
plt.title("OLAP presek — Prosečna AC_POWER (Sat × Dan)")
plt.xlabel("Dan"); plt.ylabel("Sat")
plt.xticks([0,6,13,20,30],[1,7,14,21,31])
plt.yticks([0,6,12,18,23],[0,6,12,18,23])
cbar = plt.colorbar(); cbar.set_label("Prosečna AC_POWER")
plt.tight_layout()
plt.savefig(OUT_DIR/"olap_hour_by_day.png", dpi=150, bbox_inches="tight")
plt.close()


print("\n✔ Rezultati sacuvani u 'outputs/' (model_metrics.csv, test_predictions.csv i PNG grafici).")
