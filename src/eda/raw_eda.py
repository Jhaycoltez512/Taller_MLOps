"""EDA de los CSV crudos de UNSW-NB15 (UNSW-NB15_{1..4}.csv).

Carga y limpia los cuatro archivos crudos con los nombres de NUSW-NB15_features.csv,
calcula calidad de datos, distribuciones, comportamiento temporal y correlaciones,
y escribe un reporte en Markdown con tablas y figuras:

    reports/eda_raw/README.md
    reports/eda_raw/figures/*.png
    reports/eda_raw/tables/*.csv

Uso: python -m src.eda.raw_eda [--raw data/raw] [--out reports/eda_raw] [--sample 300000]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data.clean import CATEGORICAL, ID_COLS, RAW_FILES, TARGETS, clean_raw, load_raw  # noqa: E402

SEED = 42

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------- utilidades
def md_table(df, index=True, floatfmt="{:,.4g}"):
    """DataFrame -> tabla Markdown sin dependencias extra."""
    df = df.copy()
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


class Report:
    def __init__(self, out):
        self.out = Path(out)
        self.fig_dir = self.out / "figures"
        self.tab_dir = self.out / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.parts = []

    def h(self, text, level=2):
        self.parts.append(f"{'#' * level} {text}\n")

    def p(self, text):
        self.parts.append(text.strip() + "\n")

    def table(self, df, name, index=True, floatfmt="{:,.4g}", caption=None):
        df.to_csv(self.tab_dir / f"{name}.csv", index=index)
        if caption:
            self.parts.append(f"*{caption}* (`tables/{name}.csv`)\n")
        self.parts.append(md_table(df, index=index, floatfmt=floatfmt) + "\n")

    def fig(self, fig, name, caption):
        fig.savefig(self.fig_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        self.parts.append(f"![{caption}](figures/{name}.png)\n\n*{caption}*\n")

    def write(self):
        (self.out / "README.md").write_text("\n".join(self.parts), encoding="utf-8")


# ---------------------------------------------------------------- calidad
def quality_and_clean(df):
    """Documenta las anomalías encontradas y devuelve (df_limpio, tabla_anomalías)."""
    raw_cat = df["attack_cat"]
    sport_s, dsport_s = df["sport"].astype(str).str.strip(), df["dsport"].astype(str).str.strip()
    anomalies = pd.DataFrame(
        [
            ("sport", "valor hexadecimal (0x...)", sport_s.str.startswith("0x").sum()),
            ("sport", "valor '-'", (sport_s == "-").sum()),
            ("dsport", "valor hexadecimal (0x...)", dsport_s.str.startswith("0x").sum()),
            ("dsport", "valor '-'", (dsport_s == "-").sum()),
            ("ct_ftp_cmd", "cadena en blanco ' '", (df["ct_ftp_cmd"].astype(str).str.strip() == "").sum()),
            ("ct_flw_http_mthd", "nulo", df["ct_flw_http_mthd"].isna().sum()),
            ("is_ftp_login", "nulo", df["is_ftp_login"].isna().sum()),
            ("is_ftp_login", "valor no binario (>1)", (df["is_ftp_login"] > 1).sum()),
            ("attack_cat", "nulo (registros normales)", raw_cat.isna().sum()),
            (
                "attack_cat",
                "espacios alrededor del nombre",
                (raw_cat.notna() & (raw_cat != raw_cat.str.strip())).sum(),
            ),
            ("attack_cat", "'Backdoors' (vs 'Backdoor')", (raw_cat.str.strip() == "Backdoors").sum()),
        ],
        columns=["columna", "anomalía", "registros"],
    )

    return clean_raw(df), anomalies


# ---------------------------------------------------------------- secciones
def section_overview(rep, df, feats):
    rep.h("1. Carga y esquema")
    per_file = (
        df.groupby("file")
        .agg(
            registros=("Label", "size"),
            ataques=("Label", "sum"),
            inicio=("ts", "min"),
            fin=("ts", "max"),
        )
        .assign(pct_ataque=lambda d: 100 * d["ataques"] / d["registros"])
    )
    per_file["inicio"] = per_file["inicio"].dt.strftime("%Y-%m-%d %H:%M")
    per_file["fin"] = per_file["fin"].dt.strftime("%Y-%m-%d %H:%M")
    rep.p(
        f"Los cuatro CSV crudos no tienen encabezado; los nombres de columna se toman de "
        f"`NUSW-NB15_features.csv`. En total: **{len(df):,} registros × {len(feats)} columnas** "
        f"(memoria ≈ {df.memory_usage(deep=True).sum() / 1e9:.1f} GB en pandas)."
    )
    rep.table(per_file, "registros_por_archivo", caption="Registros, ataques y rango temporal (UTC) por archivo")

    types = feats[["Name", "Type", "Description"]].copy()
    types["Name"] = types["Name"].str.strip().str.replace(" ", "")
    types["Type"] = types["Type"].str.strip().str.lower()
    types["Description"] = types["Description"].str.replace("�", "'").str.replace("?", "fl", regex=False)
    rep.p(
        "Tipos declarados: "
        + ", ".join(f"{t}: {n}" for t, n in types["Type"].value_counts().items())
        + ". El diccionario completo está en `tables/diccionario_columnas.csv`."
    )
    types.to_csv(rep.tab_dir / "diccionario_columnas.csv", index=False)


def section_quality(rep, df, anomalies):
    rep.h("2. Calidad de datos")
    rep.p("Anomalías de formato detectadas en los crudos y tratamiento aplicado en este EDA:")
    rep.table(anomalies, "anomalias", index=False)
    rep.p(
        """
Tratamiento: puertos hexadecimales convertidos a entero y `'-'` a nulo; `ct_ftp_cmd` en
blanco → nulo; `attack_cat` sin espacios, `Backdoors` → `Backdoor` y nulo → `Normal`.
`is_ftp_login` debería ser binaria pero tiene valores 2 y 4.

**Los nulos dependen de la sesión de captura, no del flujo.** En la sesión 1 las columnas
FTP/HTTP siempre están informadas (0 cuando no aplican); en la sesión 2 quedan vacías salvo
en flujos `ftp`/`http`. Es decir, nulo = "no aplica" = 0. **No** debe crearse un indicador
de nulo: identificaría la sesión 2, que tiene una tasa de ataques 10 veces mayor (fuga).
"""
    )
    nulls = pd.DataFrame(
        {
            f"nulos {c}": df.groupby("session")[c].apply(lambda x: x.isna().sum())
            for c in ("ct_ftp_cmd", "is_ftp_login", "ct_flw_http_mthd")
        }
    )
    nulls.insert(0, "registros", df.groupby("session").size())
    nulls.index.name = "sesión"
    rep.table(nulls, "nulos_por_sesion", caption="Nulos por sesión de captura")
    s2_ftp = df.loc[(df["session"] == 2) & df["ct_ftp_cmd"].notna(), "service"].value_counts()
    s2_http = df.loc[(df["session"] == 2) & df["ct_flw_http_mthd"].notna(), "service"].value_counts()
    rep.p(
        f"En la sesión 2, los valores informados de `ct_ftp_cmd` son de `service=ftp` en un "
        f"{100 * s2_ftp.get('ftp', 0) / s2_ftp.sum():.2f} % y los de `ct_flw_http_mthd` de "
        f"`service=http` en un {100 * s2_http.get('http', 0) / s2_http.sum():.1f} %."
    )

    # Consistencia label / attack_cat
    ct = pd.crosstab(df["attack_cat"] == "Normal", df["Label"])
    ct.index = pd.Index(["attack_cat ≠ Normal", "attack_cat = Normal"], name="")
    ct.columns = [f"Label={c}" for c in ct.columns]
    rep.p("Consistencia entre `Label` y `attack_cat`:")
    rep.table(ct, "consistencia_label_attack_cat")

    # Duplicados
    feat_cols = [c for c in df.columns if c not in ("file", "ts", "session")]
    dup_full = df.duplicated(subset=feat_cols)
    no_id = [c for c in feat_cols if c not in ID_COLS + TARGETS]
    dup_feat = df.duplicated(subset=no_id, keep=False)
    conflict = df[dup_feat].groupby(no_id, dropna=False)["Label"].nunique()
    n_conflict_groups = int((conflict > 1).sum())
    by_cat = (
        df.assign(dup=dup_full)
        .groupby("attack_cat")
        .agg(registros=("dup", "size"), duplicados=("dup", "sum"))
        .assign(pct_dup=lambda d: 100 * d["duplicados"] / d["registros"])
        .sort_values("registros", ascending=False)
    )
    rep.p(
        f"""
**Duplicados.** {dup_full.sum():,} filas ({100 * dup_full.mean():.1f} %) son copias exactas
de otra fila (todas las columnas, incluidos IP, puertos y timestamps). Si se ignoran
identificadores y timestamps, {dup_feat.sum():,} filas ({100 * dup_feat.mean():.1f} %)
comparten vector de features con otra, y {n_conflict_groups:,} de esos grupos tienen
**etiquetas contradictorias** (mismo vector, `Label` 0 y 1). La duplicación se concentra en
algunas categorías de ataque:
"""
    )
    rep.table(by_cat, "duplicados_por_categoria", caption="Duplicados exactos por categoría", floatfmt="{:,.1f}")
    return dup_full


def section_target(rep, df):
    rep.h("3. Variable objetivo")
    lab = df["Label"].value_counts().rename_axis("Label").to_frame("registros")
    lab["pct"] = 100 * lab["registros"] / len(df)
    cats = df["attack_cat"].value_counts().rename_axis("attack_cat").to_frame("registros")
    cats["pct_total"] = 100 * cats["registros"] / len(df)
    cats["pct_ataques"] = np.where(
        cats.index == "Normal", np.nan, 100 * cats["registros"] / (df["Label"] == 1).sum()
    )
    rep.p(
        f"""
El problema binario está **muy desbalanceado**: solo el {lab.loc[1, 'pct']:.2f} % de los
registros son ataques (≈ 1 ataque por cada {lab.loc[0, 'registros'] / lab.loc[1, 'registros']:.0f}
normales). Esto contrasta con los splits oficiales (training-set ≈ 68 % de ataques), que
fueron submuestreados. Dentro de los ataques, `Generic` y `Exploits` suman el
{cats.loc[['Generic', 'Exploits'], 'pct_ataques'].sum():.0f} %; `Worms` tiene apenas
{cats.loc['Worms', 'registros']:,} registros.
"""
    )
    rep.table(lab, "distribucion_label", floatfmt="{:,.2f}")
    rep.table(cats, "distribucion_attack_cat", floatfmt="{:,.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
    sns.barplot(x=lab.index.astype(str), y=lab["registros"], ax=axes[0], color="#4C72B0")
    axes[0].set(title="Label (0=normal, 1=ataque)", xlabel="", ylabel="registros", yscale="log")
    atk = cats.drop("Normal")
    sns.barplot(y=atk.index, x=atk["registros"], ax=axes[1], color="#C44E52")
    axes[1].set(title="Categorías de ataque", xlabel="registros (escala log)", ylabel="", xscale="log")
    for i, v in enumerate(atk["registros"]):
        axes[1].text(v, i, f" {v:,}", va="center", fontsize=9)
    rep.fig(fig, "01_objetivo", "Distribución de la etiqueta binaria y de las categorías de ataque (escala log)")


def section_temporal(rep, df):
    rep.h("4. Comportamiento temporal")
    hourly = (
        df.set_index("ts")
        .groupby([pd.Grouper(freq="h"), "Label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "normal", 1: "ataque"})
    )
    hourly = hourly[(hourly.sum(axis=1) > 0)]
    sess = df.groupby("session").agg(
        inicio=("ts", "min"), fin=("ts", "max"), registros=("Label", "size"),
        ataques=("Label", "sum"), archivos=("file", lambda f: ", ".join(sorted(f.unique()))),
    )
    sess["horas"] = (sess["fin"] - sess["inicio"]).dt.total_seconds() / 3600
    sess["pct_ataque"] = 100 * sess["ataques"] / sess["registros"]
    gap = sess["inicio"].iloc[-1] - sess["fin"].iloc[0]
    s1_atk = df.loc[(df["session"] == 1) & (df["Label"] == 1), "ts"]
    sess["inicio"] = sess["inicio"].dt.strftime("%Y-%m-%d %H:%M")
    sess["fin"] = sess["fin"].dt.strftime("%Y-%m-%d %H:%M")
    sess.index.name = "sesión"
    rep.p(
        f"""
Los timestamps (`Stime`) van de {df['ts'].min():%Y-%m-%d %H:%M} a {df['ts'].max():%Y-%m-%d %H:%M}
UTC, pero la captura **no es continua**: son {len(sess)} sesiones de simulación del IXIA
PerfectStorm separadas por {gap.days} días. Los archivos están ordenados en el tiempo y
`UNSW-NB15_2.csv` contiene el final de la sesión 1 y el comienzo de la sesión 2.
"""
    )
    rep.table(sess, "sesiones", floatfmt="{:,.2f}", caption="Sesiones de captura (corte en huecos > 6 h)")
    rep.p(
        f"""
**Cambio de prior entre sesiones**: la sesión 2 tiene un {sess['pct_ataque'].iloc[-1]:.1f} % de
ataques frente al {sess['pct_ataque'].iloc[0]:.1f} % de la sesión 1. Además, en la sesión 1 los
ataques solo ocurren entre {s1_atk.min():%H:%M} y {s1_atk.max():%H:%M} UTC
({(s1_atk.max() - s1_atk.min()).total_seconds() / 3600:.1f} h de {sess['horas'].iloc[0]:.1f} h); el resto
de la sesión es tráfico normal. En la sesión 2 los ataques se reparten de forma continua
(figura). Todas las categorías de
ataque aparecen en ambas sesiones (tabla siguiente), así que un split por sesión es viable,
pero el umbral de decisión calibrado en una sesión no se transfiere directamente a la otra.
"""
    )
    by_sess = pd.crosstab(df["attack_cat"], df["session"])
    by_sess.columns = [f"sesión {c}" for c in by_sess.columns]
    rep.table(by_sess.sort_values(by_sess.columns[0], ascending=False), "attack_cat_por_sesion",
              caption="Registros por categoría y sesión")
    sessions = [g for _, g in hourly.groupby((hourly.index.to_series().diff() > pd.Timedelta("6h")).cumsum())]
    fig, axes = plt.subplots(
        2, len(sessions), figsize=(14, 6.5), sharey="row", squeeze=False,
        gridspec_kw={"width_ratios": [len(s) for s in sessions]},
    )
    for j, s in enumerate(sessions):
        axes[0, j].plot(s.index, s["normal"], label="normal", color="#4C72B0")
        axes[0, j].plot(s.index, s["ataque"], label="ataque", color="#C44E52")
        axes[0, j].set(title=f"Sesión {j + 1}: {s.index[0]:%Y-%m-%d}", yscale="log")
        axes[1, j].plot(s.index, 100 * s["ataque"] / s.sum(axis=1), color="#8172B2")
        for ax in axes[:, j]:
            ax.tick_params(axis="x", rotation=30)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %Hh"))
    axes[0, 0].set_ylabel("registros / hora")
    axes[1, 0].set_ylabel("% ataques en la hora")
    axes[0, 0].legend()
    rep.fig(fig, "02_serie_temporal", "Registros por hora por clase y proporción de ataques (UTC)")

    cat_time = pd.crosstab(df["file"], df["attack_cat"], normalize="index") * 100
    rep.table(
        cat_time.T.sort_values(RAW_FILES[0], ascending=False),
        "attack_cat_por_archivo",
        floatfmt="{:,.3f}",
        caption="Composición (% de registros) de cada archivo por categoría",
    )
    rep.p(
        """
La proporción de ataques y la mezcla de categorías cambian entre archivos/horas: un modelo
evaluado con split aleatorio verá la misma "sesión" en train y test, lo que suele inflar
las métricas. Un split temporal (p. ej. sesión 1 → train, sesión 2 → test, o por archivo)
mide mejor la generalización.
"""
    )


def section_categorical(rep, df):
    rep.h("5. Variables categóricas")
    rep.p(
        "Cardinalidad: "
        + ", ".join(f"`{c}` = {df[c].nunique()}" for c in CATEGORICAL)
        + f"; IPs origen = {df['srcip'].nunique()}, IPs destino = {df['dstip'].nunique()}."
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, c in zip(axes, CATEGORICAL):
        g = df.groupby(c)["Label"].agg(registros="size", pct_ataque="mean")
        g["pct_ataque"] *= 100
        g = g.sort_values("registros", ascending=False).head(12)
        rep.table(g, f"categoria_{c}", floatfmt="{:,.2f}", caption=f"Top 12 valores de `{c}` y % de ataques")
        sns.barplot(y=g.index, x=g["pct_ataque"], ax=ax, color="#DD8452")
        ax.set(title=f"{c}: % ataques (top 12 por frecuencia)", xlabel="% ataques", ylabel="")
    rep.fig(fig, "03_categoricas", "Proporción de ataques por valor de proto, state y service")

    src = df.groupby("srcip")["Label"].agg(registros="size", pct_ataque="mean")
    src["pct_ataque"] *= 100
    attacker_ips = src[src["pct_ataque"] > 0]
    pure = src[(src["pct_ataque"] == 0) | (src["pct_ataque"] == 100)]
    rep.p(
        f"""
**IPs.** Solo {len(attacker_ips)} de {len(src)} IPs de origen emiten algún ataque, y
{len(pure)} IPs ({100 * pure['registros'].sum() / len(df):.1f} % de los registros) son
"puras" (0 % o 100 % de ataques). Las IPs atacantes provienen de la subred
`175.45.176.0/24` del generador de tráfico. **`srcip`, `dstip`, puertos y timestamps no deben
usarse como features**: identifican el escenario de simulación, no el comportamiento del flujo.
"""
    )
    top_src = src.sort_values("registros", ascending=False).head(15)
    rep.table(top_src, "ips_origen_top15", floatfmt="{:,.2f}", caption="Top 15 IPs de origen")


def numeric_columns(df):
    excl = set(ID_COLS + TARGETS + ["file", "ts", "session"] + CATEGORICAL)
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def section_numeric(rep, df, num):
    rep.h("6. Distribuciones de variables numéricas")
    desc = df[num].describe(percentiles=[0.5, 0.99]).T
    desc["skew"] = df[num].skew()
    desc["pct_ceros"] = 100 * (df[num] == 0).mean()
    desc["nulos"] = df[num].isna().sum()
    desc = desc.drop(columns="count")
    desc.index.name = "feature"
    rep.p(
        f"""
{len(num)} variables numéricas (se excluyen IPs, puertos, timestamps y objetivos).
La mayoría son **muy asimétricas** (skew > 10 en {(desc['skew'].abs() > 10).sum()} de ellas) y
con muchos ceros: modelos lineales/distancias requerirán `log1p` o transformaciones de rango;
árboles las toleran sin transformar.
"""
    )
    rep.table(desc, "describe_numericas", caption="Estadísticos descriptivos (todas las filas)")

    rng = np.random.default_rng(SEED)
    # Muestra estratificada para que los ataques sean visibles en los gráficos.
    idx = np.concatenate(
        [rng.choice(np.flatnonzero(df["Label"] == k), size=min(100_000, (df["Label"] == k).sum()), replace=False)
         for k in (0, 1)]
    )
    s = df.iloc[idx]
    show = ["dur", "sbytes", "dbytes", "sttl", "dttl", "Sload", "Dload", "Spkts", "Dpkts",
            "smeansz", "dmeansz", "Sintpkt", "tcprtt", "ct_srv_src", "ct_state_ttl", "ct_dst_src_ltm"]
    show = [c for c in show if c in num]
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    for ax, c in zip(axes.flat, show):
        for k, color, lbl in ((0, "#4C72B0", "normal"), (1, "#C44E52", "ataque")):
            ax.hist(np.log1p(s.loc[s["Label"] == k, c].clip(lower=0)), bins=50, alpha=0.55,
                    density=True, color=color, label=lbl)
        ax.set_title(f"log1p({c})")
    axes.flat[0].legend()
    fig.tight_layout()
    rep.fig(fig, "04_histogramas", "Distribución (log1p, densidad) de features clave por clase — muestra balanceada 100k/100k")

    fig, ax = plt.subplots(figsize=(14, 5))
    box_cols = ["sbytes", "dbytes", "Sload", "Dload", "smeansz", "dmeansz", "dur", "Sjit"]
    long = s[box_cols + ["attack_cat"]].melt(id_vars="attack_cat")
    long["value"] = np.log1p(long["value"].clip(lower=0))
    order = df["attack_cat"].value_counts().index
    sns.boxplot(data=long, x="variable", y="value", hue="attack_cat", hue_order=order,
                showfliers=False, ax=ax, linewidth=0.7)
    ax.set(ylabel="log1p(valor)", xlabel="", title="Volumen, carga y tamaño por categoría de ataque")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    rep.fig(fig, "05_boxplots_categoria", "Boxplots (sin outliers) de log1p(feature) por categoría de ataque")

    ttl = pd.crosstab(df["sttl"], df["Label"])
    ttl.columns = ["normal", "ataque"]
    ttl = ttl[ttl.sum(axis=1) > 1000].assign(pct_ataque=lambda d: 100 * d["ataque"] / (d["normal"] + d["ataque"]))
    atk_ttl = 100 * ttl.loc[ttl.index.isin([62, 254]), "ataque"].sum() / (df["Label"] == 1).sum()
    rep.p(
        f"""
**TTL.** `sttl` toma pocos valores y los ataques se concentran casi por completo en
`sttl ∈ {{62, 254}}` ({atk_ttl:.1f} % de los ataques), mientras que el tráfico normal está
sobre todo en 31 y 60. Es un artefacto de la topología del banco de pruebas (los atacantes
están a otra distancia en saltos), no una propiedad del ataque; `ct_state_ttl` lo hereda (es
la feature con mayor correlación con `Label`, ver §7). Dan métricas casi perfectas en este
dataset pero probablemente no generalizan a otra red.
"""
    )
    rep.table(ttl, "sttl_vs_label", floatfmt="{:,.2f}", caption="Valores de `sttl` con más de 1.000 registros")


def section_correlation(rep, df, num, sample):
    rep.h("7. Correlaciones")
    s = df[num + ["Label"]].sample(n=min(sample, len(df)), random_state=SEED)
    # Nulos FTP/HTTP = "no aplica" (ver §2): se imputan con 0 para correlacionar todas las filas.
    s = s.fillna({c: 0 for c in ("ct_ftp_cmd", "is_ftp_login", "ct_flw_http_mthd")})
    corr = s[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(15, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.6})
    ax.set_title(f"Correlación de Spearman (muestra aleatoria n={len(s):,})")
    rep.fig(fig, "06_correlacion_spearman", "Matriz de correlación de Spearman entre variables numéricas")

    pairs = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1)).stack()
    pairs = pairs[pairs.abs() >= 0.9].sort_values(key=np.abs, ascending=False)
    pairs = pairs.rename("spearman").rename_axis(["var_1", "var_2"]).to_frame()
    rep.p(
        f"""
Hay **{len(pairs)} pares con |ρ| ≥ 0,9**: grupos de variables redundantes, entre ellos
`is_ftp_login`/`ct_ftp_cmd`, `trans_depth`/`ct_flw_http_mthd`/`res_bdy_len`, `swin`/`dwin`,
`tcprtt = synack + ackdat`, paquetes/bytes/pérdidas (`Spkts`, `Dpkts`, `sbytes`, `dbytes`,
`sloss`, `dloss`) y duración/jitter/tiempos entre paquetes (`dur`, `Sjit`, `Djit`, `Sintpkt`,
`Dintpkt`). Los nulos FTP/HTTP se imputan con 0 para este cálculo.
Conviene eliminar o agrupar redundancias para modelos lineales y para interpretar importancias.
"""
    )
    rep.table(pairs, "pares_alta_correlacion", floatfmt="{:,.3f}", caption="Pares con |ρ de Spearman| ≥ 0,9")

    with_label = s[num].corrwith(s["Label"], method="spearman").rename("spearman_label")
    with_label = with_label.to_frame().assign(abs=lambda d: d["spearman_label"].abs())
    with_label = with_label.sort_values("abs", ascending=False).drop(columns="abs")
    with_label.index.name = "feature"
    fig, ax = plt.subplots(figsize=(8, 11))
    colors = np.where(with_label["spearman_label"] > 0, "#C44E52", "#4C72B0")
    ax.barh(with_label.index[::-1], with_label["spearman_label"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Correlación de Spearman con Label", xlabel="ρ")
    rep.fig(fig, "07_correlacion_label", "Correlación (Spearman) de cada feature con la etiqueta binaria")
    rep.table(with_label.head(15), "correlacion_con_label_top15", floatfmt="{:,.3f}",
              caption="Top 15 features por |ρ| con `Label`")
    return pairs, with_label


def section_splits(rep, df, raw):
    rep.h("8. Comparación con los splits oficiales")
    rows = []
    for name in ("training", "testing"):
        f = raw / f"UNSW_NB15_{name}-set.csv"
        if not f.exists():
            continue
        sp = pd.read_csv(f)
        rows.append((f"{name}-set oficial", len(sp), 100 * sp["label"].mean(), sp["attack_cat"].nunique()))
    rows.append(("crudos (UNSW-NB15_1..4)", len(df), 100 * df["Label"].mean(), df["attack_cat"].nunique()))
    comp = pd.DataFrame(rows, columns=["conjunto", "registros", "pct_ataque", "categorías"]).set_index("conjunto")
    rep.table(comp, "comparacion_splits", floatfmt="{:,.2f}")
    rep.p(
        """
Los splits oficiales son un **submuestreo** de los crudos (≈ 10 %) con mucha mayor proporción
de ataques, sin IPs/puertos/timestamps y con algunas features renombradas o derivadas
(`smeansz`→`smean`, `Sintpkt`→`sinpkt`, `res_bdy_len`→`response_body_len`, `rate` nueva). No
se sabe qué registros crudos componen cada split, así que no es posible hacer un split
temporal sobre ellos.
"""
    )


def section_conclusions(rep, df, dup_full, pairs, with_label):
    rate = 100 * df.groupby("session")["Label"].mean()
    rep.h("9. Conclusiones y recomendaciones")
    top = ", ".join(f"`{c}`" for c in with_label.index[:6])
    rep.p(
        f"""
1. **Limpieza obligatoria** antes de modelar: puertos hexadecimales, `ct_ftp_cmd` en blanco,
   nombres de `attack_cat` inconsistentes (`' Fuzzers '`, `Backdoors`) y `is_ftp_login` no
   binaria. Todo esto se debe implementar en un paso de preprocesamiento versionado (DVC).
   Los nulos FTP/HTTP se imputan con 0 (sin indicador de nulo: filtraría la sesión).
2. **Duplicados ({100 * dup_full.mean():.1f} % de filas exactas)**: deduplicar *antes* de
   partir en train/test; si no, copias idénticas caen en ambos lados y las métricas se inflan.
3. **Desbalance fuerte ({100 * df['Label'].mean():.1f} % de ataques)**: evaluar con PR-AUC,
   F1/recall de la clase ataque y matriz de confusión, no con accuracy; considerar
   `class_weight`/submuestreo solo en train.
4. **Fuga de información**: excluir `srcip`, `dstip`, `sport`, `dsport`, `Stime`, `Ltime` y
   `attack_cat` (para la tarea binaria). Tratar con cautela `sttl`, `dttl` y `ct_state_ttl`:
   separan casi perfectamente por artefacto del testbed. Recomendado: entrenar con y sin ellas
   y registrar ambas variantes en MLflow.
5. **Estrategia de split**: los datos tienen dos sesiones con priors muy distintos
   ({rate.iloc[0]:.1f} % vs {rate.iloc[-1]:.1f} % de ataques). Propuesta:
   - *Principal, split temporal por sesión*: sesión 1 → train/validación (estratificada por
     `Label`), sesión 2 → test. Es la evaluación más honesta (datos futuros, cambio de prior);
     reportar PR-AUC, que no depende del umbral, y analizar la recalibración del umbral.
     Ojo: los ataques de la sesión 1 se concentran en sus primeras ~2 h, por lo que la
     validación debe ser estratificada (no un corte temporal dentro de la sesión 1).
   - *Complementario, hold-out estratificado* sobre los datos deduplicados de ambas sesiones
     (p. ej. 70/15/15 estratificado por `attack_cat`), para medir la capacidad del modelo sin drift.
   - Los splits oficiales solo como referencia comparable con la literatura.
6. **Features más asociadas al ataque** (Spearman): {top}. Hay {len(pairs)} pares redundantes
   (|ρ| ≥ 0,9) que conviene podar para modelos lineales.
7. **Transformaciones**: `log1p` para bytes/cargas/tiempos (skew muy alto); one-hot o target
   encoding para `proto` (cardinalidad alta, agrupar raros), `state` y `service`.
"""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="reports/eda_raw")
    ap.add_argument("--sample", type=int, default=300_000, help="filas para la matriz de correlación")
    args = ap.parse_args()
    raw = Path(args.raw)

    print("Cargando crudos...")
    df, feats = load_raw(raw)
    df, anomalies = quality_and_clean(df)
    num = numeric_columns(df)

    rep = Report(args.out)
    rep.h("EDA de los datos crudos UNSW-NB15", level=1)
    rep.p(
        """
Reporte generado por `src/eda/raw_eda.py` sobre `data/raw/UNSW-NB15_{1..4}.csv`
(versión DVC `data/raw.dvc`). Reproducir con:

```bash
python -m src.eda.raw_eda          # escribe reports/eda_raw/{README.md,figures/,tables/}
```

Todas las cifras de este documento se calculan en el script; las tablas completas están en
`tables/` como CSV.
"""
    )
    steps = [
        ("esquema", lambda: section_overview(rep, df, feats)),
        ("calidad", lambda: section_quality(rep, df, anomalies)),
        ("objetivo", lambda: section_target(rep, df)),
        ("temporal", lambda: section_temporal(rep, df)),
        ("categóricas", lambda: section_categorical(rep, df)),
        ("numéricas", lambda: section_numeric(rep, df, num)),
        ("correlaciones", lambda: section_correlation(rep, df, num, args.sample)),
        ("splits", lambda: section_splits(rep, df, raw)),
    ]
    results = {}
    for name, fn in steps:
        print(f"  {name}...")
        results[name] = fn()
    pairs, with_label = results["correlaciones"]
    section_conclusions(rep, df, results["calidad"], pairs, with_label)
    rep.write()
    print(f"Reporte en {rep.out / 'README.md'}")


if __name__ == "__main__":
    sys.exit(main())
