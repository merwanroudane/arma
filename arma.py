import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.api import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(page_title="Économétrie de la Finance", layout="wide", initial_sidebar_state="expanded")

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 2em;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .theory-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #f0f0f0;
        border: 2px solid #333;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .example-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal
st.markdown('<p class="main-header">📊 Économétrie de la Finance</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 20px; color: #666;">Application Interactive pour Débutants</p>',
            unsafe_allow_html=True)

# Barre latérale pour la navigation
st.sidebar.title("🎯 Navigation")
chapitre = st.sidebar.radio(
    "Choisissez un chapitre:",
    ["🏠 Accueil", "📈 Chapitre 1: Processus Stationnaires & ARMA", "🔍 Chapitre 2: Tests de Stationnarité"],
    index=0
)

# ==================== PAGE D'ACCUEIL ====================
# === NEW: ADDING ATTRIBUTION TO THE SIDEBAR ===
st.sidebar.markdown("---")
st.sidebar.info("This app is created by **Dr. Hocine Belhimer**.")
if chapitre == "🏠 Accueil":
    st.markdown('<p class="sub-header" style="text-align: left;">Bienvenue dans ce cours interactif!</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="theory-box">
        <h3>📚 Objectifs du cours</h3>
        <ul>
            <li>Comprendre les séries temporelles financières</li>
            <li>Maîtriser les concepts de stationnarité</li>
            <li>Apprendre les processus ARMA</li>
            <li>Effectuer des tests de stationnarité</li>
            <li>Modéliser des données réelles</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="example-box">
        <h3>🎓 Pour qui?</h3>
        <ul>
            <li>✅ Débutants en économétrie</li>
            <li>✅ Étudiants en finance</li>
            <li>✅ Analystes financiers</li>
            <li>✅ Data scientists</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
    <h3>📖 Contenu des chapitres</h3>
    <p><strong>Chapitre 1:</strong> Introduction aux probabilités, séries temporelles, processus stationnaires, théorème de Wold, processus ARMA</p>
    <p><strong>Chapitre 2:</strong> Tests de stationnarité (Dickey-Fuller, ADF), processus ARIMA</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== CHAPITRE 1 ====================
elif chapitre == "📈 Chapitre 1: Processus Stationnaires & ARMA":

    st.sidebar.markdown("---")
    section = st.sidebar.selectbox(
        "Choisissez une section:",
        ["1.1 - Rappel de probabilité et statistiques",
         "1.2 - Séries temporelles",
         "1.3 - Processus stationnaires",
         "1.4 - Théorème de Wold",
         "1.5 - Processus ARMA"]
    )

    # ========== SECTION 1.1 ==========
    if section == "1.1 - Rappel de probabilité et statistiques":
        st.markdown('<p class="sub-header">1.1 - Rappel de Probabilité et Statistiques</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Théorie", "🎲 Variables Aléatoires", "📈 Loi Normale", "🧮 Simulations"])

        with tab1:
            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Concepts Fondamentaux</h3>
            <p>Les probabilités et statistiques sont la base de l'économétrie financière.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📌 Variable Aléatoire")
                st.markdown("""
                Une **variable aléatoire** X est une fonction qui associe un nombre réel à chaque résultat d'une expérience aléatoire.

                **Types:**
                - **Discrète**: Prend des valeurs dénombrables (ex: nombre de transactions)
                - **Continue**: Prend toutes les valeurs dans un intervalle (ex: prix d'une action)
                """)

                st.markdown("### 📐 Espérance Mathématique")
                st.latex(r"E[X] = \mu = \int_{-\infty}^{+\infty} x \cdot f(x) dx")
                st.markdown("L'espérance représente la **valeur moyenne** attendue.")

            with col2:
                st.markdown("### 📊 Variance")
                st.latex(r"Var(X) = \sigma^2 = E[(X - \mu)^2]")
                st.markdown("La variance mesure la **dispersion** autour de la moyenne.")

                st.markdown("### 📏 Écart-type")
                st.latex(r"\sigma = \sqrt{Var(X)}")
                st.markdown("L'écart-type est dans la **même unité** que X.")

        with tab2:
            st.markdown("### 🎲 Simulation de Variables Aléatoires")

            col1, col2 = st.columns([1, 2])

            with col1:
                distribution = st.selectbox(
                    "Choisissez une distribution:",
                    ["Normale", "Uniforme", "Exponentielle", "Binomiale"]
                )

                n_samples = st.slider("Nombre d'échantillons:", 100, 10000, 1000)

                if distribution == "Normale":
                    mu = st.slider("Moyenne (μ):", -10.0, 10.0, 0.0, 0.1)
                    sigma = st.slider("Écart-type (σ):", 0.1, 5.0, 1.0, 0.1)
                    data = np.random.normal(mu, sigma, n_samples)

                elif distribution == "Uniforme":
                    a = st.slider("Borne inférieure (a):", -10.0, 0.0, 0.0, 0.1)
                    b = st.slider("Borne supérieure (b):", 0.0, 10.0, 1.0, 0.1)
                    data = np.random.uniform(a, b, n_samples)

                elif distribution == "Exponentielle":
                    lambda_param = st.slider("Paramètre λ:", 0.1, 5.0, 1.0, 0.1)
                    data = np.random.exponential(1 / lambda_param, n_samples)

                else:  # Binomiale
                    n = st.slider("Nombre d'essais (n):", 1, 100, 10)
                    p = st.slider("Probabilité de succès (p):", 0.0, 1.0, 0.5, 0.01)
                    data = np.random.binomial(n, p, n_samples)

            with col2:
                # Histogramme
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    nbinsx=50,
                    name='Histogramme',
                    marker_color='lightblue',
                    opacity=0.7
                ))

                fig.update_layout(
                    title=f'Distribution {distribution}',
                    xaxis_title='Valeur',
                    yaxis_title='Fréquence',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Statistiques
                st.markdown("### 📊 Statistiques calculées:")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Moyenne empirique", f"{np.mean(data):.3f}")
                col_b.metric("Variance empirique", f"{np.var(data):.3f}")
                col_c.metric("Écart-type empirique", f"{np.std(data):.3f}")

        with tab3:
            st.markdown("### 📈 La Loi Normale (Gaussienne)")

            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Pourquoi la loi normale est-elle si importante?</h4>
            <ul>
                <li>Nombreux phénomènes naturels et financiers suivent cette loi</li>
                <li>Théorème Central Limite: somme de variables aléatoires → loi normale</li>
                <li>Base de nombreux modèles statistiques et financiers</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📐 Fonction de densité:")
            st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")

            col1, col2 = st.columns(2)

            with col1:
                mu_norm = st.slider("Moyenne μ:", -5.0, 5.0, 0.0, 0.1, key='mu_norm')
                sigma_norm = st.slider("Écart-type σ:", 0.1, 3.0, 1.0, 0.1, key='sigma_norm')

            x = np.linspace(-10, 10, 1000)
            y = stats.norm.pdf(x, mu_norm, sigma_norm)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill='tozeroy',
                name=f'N({mu_norm}, {sigma_norm}²)',
                line=dict(color='blue', width=3)
            ))

            fig.update_layout(
                title='Fonction de Densité de Probabilité',
                xaxis_title='x',
                yaxis_title='f(x)',
                height=400
            )

            with col2:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="formula-box">
            <h4>📏 Règle empirique (68-95-99.7)</h4>
            <ul>
                <li>68% des valeurs sont dans [μ - σ, μ + σ]</li>
                <li>95% des valeurs sont dans [μ - 2σ, μ + 2σ]</li>
                <li>99.7% des valeurs sont dans [μ - 3σ, μ + 3σ]</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            st.markdown("### 🧮 Théorème Central Limite - Simulation Interactive")

            st.markdown("""
            <div class="theory-box">
            <h4>🎯 Théorème Central Limite (TCL)</h4>
            <p>La somme (ou moyenne) d'un grand nombre de variables aléatoires indépendantes
            tend vers une distribution normale, quelle que soit leur distribution d'origine.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])

            with col1:
                dist_origine = st.selectbox(
                    "Distribution d'origine:",
                    ["Uniforme", "Exponentielle", "Binomiale"],
                    key='tcl_dist'
                )

                n_echantillon = st.slider("Taille de l'échantillon (n):", 2, 100, 30)
                n_simulations = st.slider("Nombre de simulations:", 100, 5000, 1000)

            # Simulation TCL
            moyennes = []
            for _ in range(n_simulations):
                if dist_origine == "Uniforme":
                    echantillon = np.random.uniform(0, 1, n_echantillon)
                elif dist_origine == "Exponentielle":
                    echantillon = np.random.exponential(1, n_echantillon)
                else:  # Binomiale
                    echantillon = np.random.binomial(10, 0.5, n_echantillon)

                moyennes.append(np.mean(echantillon))

            moyennes = np.array(moyennes)

            with col2:
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=['Distribution d\'origine', 'Distribution des moyennes'])

                # Distribution d'origine
                if dist_origine == "Uniforme":
                    sample_orig = np.random.uniform(0, 1, 1000)
                elif dist_origine == "Exponentielle":
                    sample_orig = np.random.exponential(1, 1000)
                else:
                    sample_orig = np.random.binomial(10, 0.5, 1000)

                fig.add_trace(
                    go.Histogram(x=sample_orig, nbinsx=30, name='Original', marker_color='lightcoral'),
                    row=1, col=1
                )

                # Distribution des moyennes
                fig.add_trace(
                    go.Histogram(x=moyennes, nbinsx=30, name='Moyennes', marker_color='lightblue'),
                    row=1, col=2
                )

                # Courbe normale théorique
                x_norm = np.linspace(moyennes.min(), moyennes.max(), 100)
                y_norm = stats.norm.pdf(x_norm, moyennes.mean(), moyennes.std())
                y_norm = y_norm * len(moyennes) * (moyennes.max() - moyennes.min()) / 30

                fig.add_trace(
                    go.Scatter(x=x_norm, y=y_norm, name='Loi Normale', line=dict(color='red', width=3)),
                    row=1, col=2
                )

                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            st.success(
                f"✅ La distribution des moyennes converge vers une loi normale N({moyennes.mean():.3f}, {moyennes.std():.3f}²)")

    # ========== SECTION 1.2 ==========
    elif section == "1.2 - Séries temporelles":
        st.markdown('<p class="sub-header">1.2 - Séries Temporelles</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📚 Théorie", "📈 Exemples Réels", "🎨 Composantes", "🔧 Simulations"])

        with tab1:
            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Qu'est-ce qu'une Série Temporelle?</h3>
            <p>Une <strong>série temporelle</strong> est une suite d'observations <strong>ordonnées dans le temps</strong>.</p>
            <p>Notation: {X<sub>t</sub>, t = 1, 2, ..., T} ou {X<sub>t</sub>}<sub>t∈ℤ</sub></p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 📊 Caractéristiques

                **1. Dépendance temporelle**
                - Les observations ne sont pas indépendantes
                - La valeur au temps t dépend des valeurs passées

                **2. Ordre important**
                - L'ordre des observations est crucial
                - On ne peut pas permuter les valeurs

                **3. Fréquence**
                - Secondes, minutes (trading haute fréquence)
                - Heures, jours (prix quotidiens)
                - Mois, trimestres, années (données macro)
                """)

            with col2:
                st.markdown("""
                ### 💼 Applications en Finance

                - 📈 **Prix des actions**: cours quotidiens
                - 💱 **Taux de change**: EUR/USD, etc.
                - 📊 **Indices boursiers**: CAC 40, S&P 500
                - 💰 **Taux d'intérêt**: LIBOR, OIS
                - 📉 **Volatilité**: VIX, mesures GARCH
                - 🏢 **Données macro**: PIB, inflation, chômage
                """)

            st.markdown("### 📐 Notation Mathématique")

            st.latex(r"X_t = \mu + \epsilon_t")
            st.markdown("où:")
            st.latex(r"\mu = \text{tendance ou niveau moyen}")
            st.latex(r"\epsilon_t = \text{composante aléatoire (bruit)}")

        with tab2:
            st.markdown("### 📈 Exemples de Séries Temporelles Financières")

            type_serie = st.selectbox(
                "Choisissez un type de série:",
                ["Prix d'action (avec tendance)", "Rendements (stationnaire)",
                 "Taux d'intérêt", "Indice boursier"]
            )

            n_points = st.slider("Nombre d'observations:", 50, 500, 250)

            t = np.arange(n_points)

            if type_serie == "Prix d'action (avec tendance)":
                # Mouvement brownien géométrique
                mu_drift = 0.0005
                sigma = 0.02
                S0 = 100
                rendements = np.random.normal(mu_drift, sigma, n_points)
                prix = S0 * np.exp(np.cumsum(rendements))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=prix, mode='lines', name='Prix', line=dict(color='blue', width=2)))
                fig.update_layout(
                    title='Simulation de Prix d\'Action (Mouvement Brownien Géométrique)',
                    xaxis_title='Temps (jours)',
                    yaxis_title='Prix (€)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation</h4>
                <ul>
                    <li>📈 Tendance haussière (drift positif)</li>
                    <li>🎲 Fluctuations aléatoires (volatilité)</li>
                    <li>❌ <strong>Non-stationnaire</strong>: la moyenne évolue dans le temps</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            elif type_serie == "Rendements (stationnaire)":
                # Rendements quotidiens
                mu_rend = 0.0005
                sigma_rend = 0.02
                rendements = np.random.normal(mu_rend, sigma_rend, n_points)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=t, y=rendements, mode='lines', name='Rendements', line=dict(color='green', width=1)))
                fig.add_hline(y=mu_rend, line_dash="dash", line_color="red", annotation_text="Moyenne")
                fig.add_hline(y=mu_rend + 2 * sigma_rend, line_dash="dot", line_color="orange")
                fig.add_hline(y=mu_rend - 2 * sigma_rend, line_dash="dot", line_color="orange")

                fig.update_layout(
                    title='Rendements Quotidiens',
                    xaxis_title='Temps (jours)',
                    yaxis_title='Rendement',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation</h4>
                <ul>
                    <li>📊 Moyenne constante autour de 0.05%</li>
                    <li>📏 Variance constante</li>
                    <li>✅ <strong>Stationnaire</strong>: propriétés statistiques constantes</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            elif type_serie == "Taux d'intérêt":
                # Processus de retour à la moyenne (Ornstein-Uhlenbeck)
                theta = 0.1  # vitesse de retour
                mu_taux = 0.03  # niveau moyen
                sigma_taux = 0.005
                taux = np.zeros(n_points)
                taux[0] = mu_taux

                for i in range(1, n_points):
                    taux[i] = taux[i - 1] + theta * (mu_taux - taux[i - 1]) + sigma_taux * np.random.normal()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=taux * 100, mode='lines', name='Taux d\'intérêt',
                                         line=dict(color='purple', width=2)))
                fig.add_hline(y=mu_taux * 100, line_dash="dash", line_color="red", annotation_text="Niveau moyen")

                fig.update_layout(
                    title='Simulation de Taux d\'Intérêt (Modèle de Vasicek)',
                    xaxis_title='Temps',
                    yaxis_title='Taux (%)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation</h4>
                <ul>
                    <li>🔄 <strong>Retour à la moyenne</strong>: le taux revient vers 3%</li>
                    <li>📊 Utilisé pour modéliser les taux d'intérêt</li>
                    <li>✅ Propriété de stationnarité</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:  # Indice boursier
                # Indice avec saisonnalité et tendance
                tendance = 0.001 * t
                saisonnalite = 5 * np.sin(2 * np.pi * t / 50)
                bruit = np.random.normal(0, 2, n_points)
                indice = 100 + tendance + saisonnalite + bruit

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=t, y=indice, mode='lines', name='Indice', line=dict(color='darkblue', width=2)))
                fig.add_trace(go.Scatter(x=t, y=100 + tendance, mode='lines', name='Tendance',
                                         line=dict(color='red', dash='dash')))

                fig.update_layout(
                    title='Indice Boursier avec Tendance et Cycles',
                    xaxis_title='Temps',
                    yaxis_title='Valeur de l\'indice',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation</h4>
                <ul>
                    <li>📈 Tendance croissante long terme</li>
                    <li>🔄 Cycles économiques (saisonnalité)</li>
                    <li>🎲 Fluctuations aléatoires court terme</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### 🎨 Décomposition d'une Série Temporelle")

            st.markdown("""
            <div class="theory-box">
            <h4>📐 Modèle de décomposition</h4>
            <p>Une série temporelle peut être décomposée en plusieurs composantes:</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"X_t = T_t + S_t + C_t + I_t")

            st.markdown("""
            - **T_t**: Tendance (Trend) - mouvement long terme
            - **S_t**: Saisonnalité (Seasonality) - variations périodiques
            - **C_t**: Cycle - fluctuations moyen terme
            - **I_t**: Irrégularité (Irregular) - composante aléatoire
            """)

            # Simulation interactive
            st.markdown("### 🎛️ Créez votre propre série temporelle")

            col1, col2, col3 = st.columns(3)

            with col1:
                tendance_coef = st.slider("Force de la tendance:", 0.0, 0.5, 0.1, 0.01)
            with col2:
                saison_amp = st.slider("Amplitude saisonnalité:", 0.0, 20.0, 10.0, 1.0)
            with col3:
                bruit_std = st.slider("Intensité du bruit:", 0.0, 10.0, 3.0, 0.5)

            t = np.arange(200)

            # Composantes
            tendance = tendance_coef * t
            saisonnalite = saison_amp * np.sin(2 * np.pi * t / 25)
            cycle = 5 * np.sin(2 * np.pi * t / 50)
            irregulier = np.random.normal(0, bruit_std, len(t))

            serie = 100 + tendance + saisonnalite + cycle + irregulier

            # Graphique de décomposition
            fig = make_subplots(
                rows=5, cols=1,
                subplot_titles=['Série Complète', 'Tendance', 'Saisonnalité', 'Cycle', 'Irrégularité'],
                vertical_spacing=0.05
            )

            fig.add_trace(go.Scatter(x=t, y=serie, mode='lines', name='Série', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=100 + tendance, mode='lines', name='Tendance', line=dict(color='red')),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=saisonnalite, mode='lines', name='Saisonnalité', line=dict(color='blue')),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=cycle, mode='lines', name='Cycle', line=dict(color='green')), row=4, col=1)
            fig.add_trace(go.Scatter(x=t, y=irregulier, mode='lines', name='Irrégularité', line=dict(color='orange')),
                          row=5, col=1)

            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### 🔧 Simulateur de Processus Stochastiques")

            processus_type = st.selectbox(
                "Type de processus:",
                ["Bruit Blanc", "Marche Aléatoire", "Processus AR(1)", "Processus MA(1)"]
            )

            n_obs = st.slider("Nombre d'observations:", 100, 1000, 500, key='sim_obs')

            if processus_type == "Bruit Blanc":
                st.markdown("""
                <div class="formula-box">
                <h4>📐 Bruit Blanc (White Noise)</h4>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)")
                st.markdown("</div>", unsafe_allow_html=True)

                sigma_bb = st.slider("Écart-type σ:", 0.1, 5.0, 1.0, 0.1)
                serie_sim = np.random.normal(0, sigma_bb, n_obs)
                titre = f"Bruit Blanc N(0, {sigma_bb}²)"

            elif processus_type == "Marche Aléatoire":
                st.markdown("""
                <div class="formula-box">
                <h4>📐 Marche Aléatoire (Random Walk)</h4>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = X_{t-1} + \epsilon_t")
                st.markdown("</div>", unsafe_allow_html=True)

                sigma_rw = st.slider("Écart-type σ:", 0.1, 5.0, 1.0, 0.1)
                innovations = np.random.normal(0, sigma_rw, n_obs)
                serie_sim = np.cumsum(innovations)
                titre = "Marche Aléatoire"

            elif processus_type == "Processus AR(1)":
                st.markdown("""
                <div class="formula-box">
                <h4>📐 Processus Autorégressif d'ordre 1</h4>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \phi X_{t-1} + \epsilon_t")
                st.markdown("</div>", unsafe_allow_html=True)

                phi = st.slider("Coefficient φ:", -0.99, 0.99, 0.7, 0.01)
                sigma_ar = st.slider("Écart-type σ:", 0.1, 5.0, 1.0, 0.1)

                serie_sim = np.zeros(n_obs)
                for t in range(1, n_obs):
                    serie_sim[t] = phi * serie_sim[t - 1] + np.random.normal(0, sigma_ar)

                titre = f"Processus AR(1), φ={phi}"

                if abs(phi) >= 1:
                    st.warning("⚠️ Processus non-stationnaire (|φ| ≥ 1)")
                else:
                    st.success(f"✅ Processus stationnaire (|φ| < 1)")

            else:  # MA(1)
                st.markdown("""
                <div class="formula-box">
                <h4>📐 Processus Moyenne Mobile d'ordre 1</h4>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \epsilon_t + \theta \epsilon_{t-1}")
                st.markdown("</div>", unsafe_allow_html=True)

                theta = st.slider("Coefficient θ:", -0.99, 0.99, 0.5, 0.01)
                sigma_ma = st.slider("Écart-type σ:", 0.1, 5.0, 1.0, 0.1)

                epsilon = np.random.normal(0, sigma_ma, n_obs)
                serie_sim = np.zeros(n_obs)
                serie_sim[0] = epsilon[0]
                for t in range(1, n_obs):
                    serie_sim[t] = epsilon[t] + theta * epsilon[t - 1]

                titre = f"Processus MA(1), θ={theta}"

            # Graphiques
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(y=serie_sim, mode='lines', name=titre, line=dict(width=1.5)))
                fig1.update_layout(title=titre, xaxis_title='Temps', yaxis_title='Valeur', height=300)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=serie_sim, nbinsx=50, name='Distribution'))
                fig2.update_layout(title='Distribution', xaxis_title='Valeur', yaxis_title='Fréquence', height=300)
                st.plotly_chart(fig2, use_container_width=True)

            # Statistiques
            st.markdown("### 📊 Statistiques de la série simulée")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Moyenne", f"{np.mean(serie_sim):.4f}")
            col_b.metric("Écart-type", f"{np.std(serie_sim):.4f}")
            col_c.metric("Minimum", f"{np.min(serie_sim):.4f}")
            col_d.metric("Maximum", f"{np.max(serie_sim):.4f}")

    # ========== SECTION 1.3 ==========
    elif section == "1.3 - Processus stationnaires":
        st.markdown('<p class="sub-header">1.3 - Processus Stationnaires</p>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📚 Théorie", "🔄 Stationnarité Forte vs Faible", "🎯 Applications"])

        with tab1:
            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Qu'est-ce que la Stationnarité?</h3>
            <p>Un processus stochastique est <strong>stationnaire</strong> si ses propriétés statistiques
            <strong>ne changent pas dans le temps</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📐 Stationnarité Forte (Stricte)")
                st.markdown("""
                <div class="formula-box">
                <p>La distribution conjointe de (X<sub>t₁</sub>, ..., X<sub>tₙ</sub>) est la même que
                celle de (X<sub>t₁+h</sub>, ..., X<sub>tₙ+h</sub>) pour tout h.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"F(x_1, ..., x_n; t_1, ..., t_n) = F(x_1, ..., x_n; t_1+h, ..., t_n+h)")

                st.markdown("""
                **Implications:**
                - Toute la distribution reste identique
                - Condition très forte, rarement vérifiable
                """)

            with col2:
                st.markdown("### 📊 Stationnarité Faible (au second ordre)")
                st.markdown("""
                <div class="formula-box">
                <p>Les deux premiers moments sont constants dans le temps.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Trois conditions:**")
                st.latex(r"1. \quad E[X_t] = \mu \quad \forall t")
                st.latex(r"2. \quad Var(X_t) = \sigma^2 \quad \forall t")
                st.latex(r"3. \quad Cov(X_t, X_{t+h}) = \gamma(h) \quad \text{(dépend seulement de h)}")

                st.markdown("""
                **C'est la définition utilisée en pratique!**
                """)

            st.markdown("---")

            st.markdown("### 🔑 Concepts Clés")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="example-box">
                <h4>📊 Moyenne Constante</h4>
                <p>E[X<sub>t</sub>] = μ</p>
                <p>La série oscille autour d'un niveau fixe</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="example-box">
                <h4>📏 Variance Constante</h4>
                <p>Var(X<sub>t</sub>) = σ²</p>
                <p>La dispersion ne change pas</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="example-box">
                <h4>🔄 Autocovariance</h4>
                <p>γ(h) = Cov(X<sub>t</sub>, X<sub>t+h</sub>)</p>
                <p>Dépend seulement du décalage h</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📐 Fonction d'Autocovariance")
            st.latex(r"\gamma(h) = Cov(X_t, X_{t+h}) = E[(X_t - \mu)(X_{t+h} - \mu)]")

            st.markdown("**Propriétés:**")
            st.latex(r"\gamma(0) = Var(X_t) = \sigma^2")
            st.latex(r"\gamma(h) = \gamma(-h) \quad \text{(symétrie)}")
            st.latex(r"|\gamma(h)| \leq \gamma(0)")

            st.markdown("### 📈 Fonction d'Autocorrélation (ACF)")
            st.latex(r"\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{Cov(X_t, X_{t+h})}{Var(X_t)}")

            st.markdown("**Propriétés:**")
            st.latex(r"-1 \leq \rho(h) \leq 1")
            st.latex(r"\rho(0) = 1")

        with tab2:
            st.markdown("### 🔄 Comparaison: Processus Stationnaire vs Non-Stationnaire")

            n_points = 300
            t = np.arange(n_points)

            # Processus stationnaire (AR(1) avec |φ| < 1)
            phi_stat = 0.7
            stationnaire = np.zeros(n_points)
            for i in range(1, n_points):
                stationnaire[i] = phi_stat * stationnaire[i - 1] + np.random.normal(0, 1)

            # Processus non-stationnaire (marche aléatoire)
            non_stationnaire = np.cumsum(np.random.normal(0, 1, n_points))

            # Tendance déterministe
            tendance = 0.05 * t + np.random.normal(0, 1, n_points)

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['✅ Stationnaire (AR)', '❌ Non-Stationnaire (Marche Aléatoire)',
                                '❌ Tendance Déterministe']
            )

            fig.add_trace(go.Scatter(y=stationnaire, mode='lines', name='Stationnaire', line=dict(color='green')),
                          row=1, col=1)
            fig.add_trace(go.Scatter(y=non_stationnaire, mode='lines', name='Non-Stat', line=dict(color='red')), row=1,
                          col=2)
            fig.add_trace(go.Scatter(y=tendance, mode='lines', name='Tendance', line=dict(color='orange')), row=1,
                          col=3)

            # Lignes de moyenne
            fig.add_hline(y=np.mean(stationnaire), line_dash="dash", line_color="green", row=1, col=1)

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="example-box">
                <h4>✅ Stationnaire</h4>
                <ul>
                    <li>Moyenne constante ≈ 0</li>
                    <li>Variance constante</li>
                    <li>Revient vers la moyenne</li>
                    <li>Prévisible à long terme</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="warning-box">
                <h4>❌ Non-Stationnaire</h4>
                <ul>
                    <li>Pas de niveau moyen fixe</li>
                    <li>Variance croissante</li>
                    <li>Drift aléatoire</li>
                    <li>Imprévisible long terme</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="warning-box">
                <h4>❌ Avec Tendance</h4>
                <ul>
                    <li>Moyenne croissante</li>
                    <li>Tendance déterministe</li>
                    <li>Nécessite différenciation</li>
                    <li>Ou détrend</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("### 📊 Fonction d'Autocorrélation (ACF)")

            # Calcul ACF
            from statsmodels.tsa.stattools import acf

            acf_stat = acf(stationnaire, nlags=40)
            acf_nonstat = acf(non_stationnaire, nlags=40)

            fig2 = make_subplots(rows=1, cols=2, subplot_titles=['ACF - Stationnaire', 'ACF - Non-Stationnaire'])

            fig2.add_trace(go.Bar(y=acf_stat, name='ACF Stat', marker_color='green'), row=1, col=1)
            fig2.add_trace(go.Bar(y=acf_nonstat, name='ACF Non-Stat', marker_color='red'), row=1, col=2)

            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("""
            <div class="theory-box">
            <h4>💡 Interprétation de l'ACF</h4>
            <ul>
                <li><strong>Stationnaire:</strong> ACF décroît rapidement vers 0</li>
                <li><strong>Non-Stationnaire:</strong> ACF décroît très lentement, reste élevée</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### 🎯 Importance de la Stationnarité en Finance")

            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Pourquoi la stationnarité est-elle importante?</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### ✅ Avantages d'un processus stationnaire

                1. **Prévision possible**
                   - Le comportement passé est représentatif du futur
                   - Modèles statistiques applicables

                2. **Théorie asymptotique valide**
                   - Convergence des estimateurs
                   - Tests statistiques fiables

                3. **Interprétation stable**
                   - Paramètres constants dans le temps
                   - Relations économiques stables

                4. **Modélisation ARMA applicable**
                   - Base théorique solide
                   - Estimation cohérente
                """)

            with col2:
                st.markdown("""
                ### ⚠️ Problèmes avec la non-stationnarité

                1. **Régression fallacieuse (spurious)**
                   - Corrélations artificielles
                   - R² élevé mais non significatif

                2. **Tests invalides**
                   - Distributions non-standard
                   - Inférence incorrecte

                3. **Prévisions non fiables**
                   - Variance croissante
                   - Intervalles de confiance inexacts

                4. **Nécessité de transformation**
                   - Différenciation
                   - Détrend
                   - Transformation logarithmique
                """)

            st.markdown("---")

            st.markdown("### 💼 Applications Pratiques")

            st.markdown("""
            <div class="example-box">
            <h4>📈 Séries Financières Typiques</h4>

            **Généralement NON-stationnaires:**
            - Prix des actions (marche aléatoire)
            - Indices boursiers (tendance)
            - PIB (croissance tendancielle)
            - Taux de change (drift)

            **Généralement stationnaires:**
            - Rendements d'actions
            - Variations de taux d'intérêt
            - Spread de taux
            - Primes de risque
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔧 Comment rendre une série stationnaire?")

            option_transfo = st.selectbox(
                "Choisissez une transformation:",
                ["Différenciation première", "Différenciation logarithmique (rendements)", "Détrend (régression)"]
            )

            # Série non-stationnaire simulée (prix)
            np.random.seed(42)
            n = 200
            prix = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))

            if option_transfo == "Différenciation première":
                transformee = np.diff(prix)
                titre_transfo = "Différence Première: ΔX_t = X_t - X_{t-1}"

            elif option_transfo == "Différenciation logarithmique (rendements)":
                transformee = np.diff(np.log(prix))
                titre_transfo = "Rendements: r_t = ln(P_t) - ln(P_{t-1})"

            else:  # Détrend
                from scipy import signal

                transformee = signal.detrend(prix)
                titre_transfo = "Série détendue (régression linéaire)"

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=['❌ Série Originale (Non-Stationnaire)', f'✅ {titre_transfo}'])

            fig.add_trace(go.Scatter(y=prix, mode='lines', name='Prix', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(y=transformee, mode='lines', name='Transformée', line=dict(color='green')), row=1,
                          col=2)

            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Statistiques comparatives
            st.markdown("### 📊 Statistiques Comparatives")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Série Originale:**")
                st.write(f"- Moyenne: {np.mean(prix):.2f}")
                st.write(f"- Écart-type: {np.std(prix):.2f}")
                st.write(f"- Min: {np.min(prix):.2f}")
                st.write(f"- Max: {np.max(prix):.2f}")

            with col_b:
                st.markdown("**Série Transformée:**")
                st.write(f"- Moyenne: {np.mean(transformee):.6f}")
                st.write(f"- Écart-type: {np.std(transformee):.6f}")
                st.write(f"- Min: {np.min(transformee):.6f}")
                st.write(f"- Max: {np.max(transformee):.6f}")

    # ========== SECTION 1.4 ==========
    elif section == "1.4 - Théorème de Wold":
        st.markdown('<p class="sub-header">1.4 - Théorème de Wold (Décomposition de Wold)</p>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📚 Théorème", "🔍 Démonstration Visuelle", "💡 Applications"])

        with tab1:
            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Le Théorème de Wold</h3>
            <p>Tout processus stationnaire peut être décomposé en deux parties indépendantes:</p>
            <ol>
                <li>Une partie <strong>stochastique</strong> (imprévisible)</li>
                <li>Une partie <strong>déterministe</strong> (prévisible)</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📐 Formulation Mathématique")

            st.latex(r"X_t = \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j} + V_t")

            st.markdown(r"""
            où:
            - **Partie stochastique:** $\sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}$ avec $\epsilon_t$ un bruit blanc
            - **Partie déterministe:** $V_t$ (prévisible à partir du passé infini)
            - **Coefficients:** $\psi_0 = 1$ et $\sum_{j=0}^{\infty} \psi_j^2 < \infty$
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="formula-box">
                <h4>📊 Partie Stochastique (MA(∞))</h4>
                """, unsafe_allow_html=True)
                st.latex(r"\sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}")
                st.markdown("""
                <ul>
                    <li>Combinaison linéaire infinie de chocs aléatoires</li>
                    <li>Représentation MA(∞)</li>
                    <li>Innovation: ε<sub>t</sub> ~ BB(0, σ²)</li>
                    <li>Non prévisible (nouvelle information)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="example-box">
                <h4>🎯 Partie Déterministe</h4>
                <p>V<sub>t</sub></p>
                <ul>
                    <li>Parfaitement prévisible</li>
                    <li>Fonctions déterministes du temps</li>
                    <li>Exemples: constantes, tendances, saisonnalités</li>
                    <li>Indépendante des innovations</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("### 🔑 Implications Importantes")

            st.markdown("""
            <div class="theory-box">
            <h4>💡 Conséquences du Théorème de Wold</h4>

            1. **Tout processus stationnaire a une représentation MA(∞)**
               - Fondement théorique des modèles ARMA
               - Justification de l'approche Box-Jenkins

            2. **Séparation entre aléatoire et déterministe**
               - Permet d'isoler la partie imprévisible
               - Facilite la modélisation

            3. **Base des prévisions**
               - Prévisions optimales = partie déterministe
               - Erreur de prévision = innovations futures

            4. **Condition de sommabilité**
               - $\sum_{j=0}^{\infty} \psi_j^2 < \infty$ garantit la variance finie
               - Assure la stationnarité
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📈 Représentation Graphique")

            st.image(
                "https://via.placeholder.com/800x300/e8f4f8/1f77b4?text=X_t+%3D+Partie+Stochastique+%2B+Partie+Déterministe",
                use_container_width=True)

        with tab2:
            st.markdown("### 🔍 Démonstration Visuelle Interactive")

            st.markdown("""
            <div class="example-box">
            <h4>🎨 Construisons un processus selon Wold</h4>
            <p>Expérimentez avec les différentes composantes!</p>
            </div>
            """, unsafe_allow_html=True)

            # Contrôles
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Partie Stochastique (MA)**")
                n_coeffs = st.slider("Nombre de coefficients ψ:", 1, 20, 5)
                sigma_innov = st.slider("σ (innovations):", 0.1, 3.0, 1.0, 0.1)

            with col2:
                st.markdown("**Partie Déterministe**")
                tendance = st.checkbox("Ajouter tendance")
                if tendance:
                    trend_coef = st.slider("Coefficient tendance:", -0.1, 0.1, 0.01, 0.01)

                saisonalite = st.checkbox("Ajouter saisonnalité")
                if saisonalite:
                    amp_saison = st.slider("Amplitude saisonnalité:", 0.0, 10.0, 5.0, 0.5)

            with col3:
                st.markdown("**Paramètres généraux**")
                n_obs = st.slider("Nombre observations:", 100, 500, 250)
                graine = st.number_input("Graine aléatoire:", 0, 1000, 42)

            # Simulation
            np.random.seed(graine)

            # Coefficients MA qui décroissent
            psi = np.array([0.9 ** j for j in range(n_coeffs)])

            # Innovations (bruit blanc)
            epsilon = np.random.normal(0, sigma_innov, n_obs + n_coeffs)

            # Partie stochastique (MA)
            stochastique = np.zeros(n_obs)
            for t in range(n_obs):
                stochastique[t] = np.sum(psi * epsilon[t:t + n_coeffs])

            # Partie déterministe
            deterministe = np.zeros(n_obs)
            t_array = np.arange(n_obs)

            if tendance:
                deterministe += trend_coef * t_array

            if saisonalite:
                deterministe += amp_saison * np.sin(2 * np.pi * t_array / 25)

            # Processus complet
            X_t = stochastique + deterministe

            # Graphiques
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Partie Stochastique (MA)', 'Partie Déterministe', 'Processus Complet (Wold)'],
                vertical_spacing=0.1
            )

            fig.add_trace(go.Scatter(y=stochastique, mode='lines', name='Stochastique', line=dict(color='blue')), row=1,
                          col=1)
            fig.add_trace(go.Scatter(y=deterministe, mode='lines', name='Déterministe', line=dict(color='red')), row=2,
                          col=1)
            fig.add_trace(go.Scatter(y=X_t, mode='lines', name='X_t', line=dict(color='black', width=2)), row=3, col=1)

            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Affichage des coefficients
            st.markdown("### 📊 Coefficients MA (ψ)")

            fig_coef = go.Figure()
            fig_coef.add_trace(go.Bar(x=list(range(n_coeffs)), y=psi, name='ψ_j', marker_color='lightblue'))
            fig_coef.update_layout(
                title='Décroissance des coefficients ψ',
                xaxis_title='j (décalage)',
                yaxis_title='ψ_j',
                height=300
            )
            st.plotly_chart(fig_coef, use_container_width=True)

            st.markdown(f"""
            <div class="formula-box">
            <h4>✅ Vérification de la condition de sommabilité</h4>
            <p>Somme des carrés: Σψ²<sub>j</sub> = {np.sum(psi ** 2):.4f} < ∞</p>
            <p>✅ La condition est satisfaite!</p>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### 💡 Applications et Conséquences")

            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Pourquoi le Théorème de Wold est-il fondamental?</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 📈 En Théorie

                1. **Fondement des modèles ARMA**
                   - Tout processus stationnaire → MA(∞)
                   - Approximation par ARMA fini
                   - Justification théorique solide

                2. **Représentation universelle**
                   - Unicité de la décomposition
                   - Innovations orthogonales
                   - Identification claire

                3. **Théorie des prévisions**
                   - Prévision optimale = projection
                   - Erreur = innovations futures
                   - Variance de prévision calculable
                """)

            with col2:
                st.markdown("""
                ### 💼 En Pratique

                1. **Modélisation financière**
                   - Rendements d'actifs
                   - Volatilité (GARCH)
                   - Taux d'intérêt

                2. **Choix du modèle**
                   - AR, MA, ou ARMA?
                   - Critères d'information (AIC, BIC)
                   - Parcimonie vs précision

                3. **Diagnostic**
                   - Analyse des résidus
                   - Test de bruit blanc
                   - Validation du modèle
                """)

            st.markdown("---")

            st.markdown("### 🔄 Du Théorème de Wold aux Modèles ARMA")

            st.markdown("""
            <div class="example-box">
            <h4>🎓 Cheminement logique</h4>

            1. **Théorème de Wold** → Tout processus stationnaire a une représentation MA(∞)

            2. **Problème pratique** → Infinité de paramètres à estimer!

            3. **Solution: Parcimonie** → Approximation par modèles finis

            4. **Modèles ARMA** → Représentation compacte et efficace

            5. **AR(p)**: p paramètres au lieu de ∞

            6. **MA(q)**: q paramètres au lieu de ∞

            7. **ARMA(p,q)**: p+q paramètres, maximum de flexibilité
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Exemple Concret: Du MA(∞) au MA(2)")

            # Simulation comparative
            np.random.seed(123)
            n = 300
            epsilon = np.random.normal(0, 1, n + 50)

            # MA(infini) - 50 coefficients
            psi_inf = np.array([0.8 ** j for j in range(50)])
            ma_inf = np.zeros(n)
            for t in range(n):
                ma_inf[t] = np.sum(psi_inf * epsilon[t:t + 50])

            # MA(2) - approximation
            psi_2 = np.array([1, 0.8, 0.64])  # ψ0=1, ψ1=0.8, ψ2=0.64
            ma_2 = np.zeros(n)
            for t in range(n):
                ma_2[t] = np.sum(psi_2 * epsilon[t:t + 3])

            fig_comp = make_subplots(rows=2, cols=1,
                                     subplot_titles=['MA(∞) - 50 coefficients', 'MA(2) - Approximation'])

            fig_comp.add_trace(go.Scatter(y=ma_inf, mode='lines', name='MA(∞)', line=dict(color='blue')), row=1, col=1)
            fig_comp.add_trace(go.Scatter(y=ma_2, mode='lines', name='MA(2)', line=dict(color='red')), row=2, col=1)

            fig_comp.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Corrélation
            correlation = np.corrcoef(ma_inf, ma_2)[0, 1]

            st.markdown(f"""
            <div class="formula-box">
            <h4>📊 Qualité de l'approximation</h4>
            <p>Corrélation entre MA(∞) et MA(2): <strong>{correlation:.4f}</strong></p>
            <p>{"✅ Excellente approximation!" if correlation > 0.95 else "⚠️ Approximation acceptable"}</p>
            </div>
            """, unsafe_allow_html=True)

    # ========== SECTION 1.5 ==========
    elif section == "1.5 - Processus ARMA":
        st.markdown('<p class="sub-header">1.5 - Processus ARMA (AutoRegressive Moving Average)</p>',
                    unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 Introduction",
            "📊 Processus AR(p)",
            "📈 Processus MA(q)",
            "🔄 Processus ARMA(p,q)",
            "🎯 Application Pratique"
        ])

        with tab1:
            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Les Modèles ARMA : Vue d'Ensemble</h3>
            <p>Les modèles ARMA combinent deux approches complémentaires pour modéliser les séries temporelles stationnaires.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="formula-box">
                <h4>📊 AR - AutoRégressif</h4>
                <p>Le présent dépend du <strong>passé de la série</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t")
                st.markdown("""
                <p><strong>Mémoire longue</strong> via les valeurs passées</p>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="example-box">
                <h4>📈 MA - Moyenne Mobile</h4>
                <p>Le présent dépend des <strong>erreurs passées</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t")
                st.markdown("""
                <p><strong>Mémoire courte</strong> via les chocs passés</p>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="theory-box">
                <h4>🔄 ARMA - Combinaison</h4>
                <p><strong>Le meilleur des deux mondes</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.latex(r"X_t = \sum_{i=1}^p \phi_i X_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t")
                st.markdown("""
                <p><strong>Flexibilité maximale</strong></p>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("### 📐 Notation Opérateur de Retard (Backshift)")

            st.markdown("""
            <div class="formula-box">
            <h4>🔧 Opérateur L (Lag Operator)</h4>
            """, unsafe_allow_html=True)

            st.latex(r"L \cdot X_t = X_{t-1}")
            st.latex(r"L^k \cdot X_t = X_{t-k}")

            st.markdown("""
            **Polynômes caractéristiques:**
            """)

            st.latex(r"\Phi(L) = 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p")
            st.latex(r"\Theta(L) = 1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q")

            st.markdown("""
            **Forme compacte ARMA:**
            """)

            st.latex(r"\Phi(L) X_t = \Theta(L) \epsilon_t")

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 🔑 Concepts Fondamentaux")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="theory-box">
                <h4>✅ Stationnarité</h4>
                <p><strong>Condition:</strong> Les racines de Φ(L) = 0 doivent être <strong>hors du cercle unité</strong></p>
                <p>|z| > 1 pour toutes les racines z</p>
                <p>Pour AR(1): |φ| < 1</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="example-box">
                <h4>🔄 Inversibilité</h4>
                <p><strong>Condition:</strong> Les racines de Θ(L) = 0 doivent être <strong>hors du cercle unité</strong></p>
                <p>Permet de représenter ARMA comme AR(∞)</p>
                <p>Pour MA(1): |θ| < 1</p>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 📊 Processus AutoRégressif AR(p)")

            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Définition</h3>
            <p>Un processus AR(p) est un processus où la valeur actuelle dépend linéairement de ses p valeurs passées.</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t")

            st.markdown("""
            où:
            - c est une constante
            - φ₁, φ₂, ..., φₚ sont les coefficients autorégressifs
            - εₜ ~ BB(0, σ²) est un bruit blanc
            """)

            # Sélection de l'ordre
            ordre_ar = st.selectbox("Choisissez l'ordre du processus AR:", ["AR(1)", "AR(2)", "AR(3)"])

            if ordre_ar == "AR(1)":
                st.markdown("### 📐 Processus AR(1)")

                st.latex(r"X_t = \phi X_{t-1} + \epsilon_t")

                col1, col2 = st.columns([1, 2])

                with col1:
                    phi_ar1 = st.slider("Coefficient φ:", -0.99, 0.99, 0.7, 0.01, key='phi_ar1')
                    sigma_ar1 = st.slider("σ (bruit blanc):", 0.1, 3.0, 1.0, 0.1, key='sigma_ar1')
                    n_ar1 = st.slider("Nombre observations:", 100, 500, 250, key='n_ar1')

                    if abs(phi_ar1) < 1:
                        st.success(f"✅ Processus stationnaire (|φ| = {abs(phi_ar1):.2f} < 1)")
                    else:
                        st.error(f"❌ Processus non-stationnaire (|φ| = {abs(phi_ar1):.2f} ≥ 1)")

                    # Propriétés théoriques
                    if abs(phi_ar1) < 1:
                        st.markdown("**Propriétés théoriques:**")
                        mu_theor = 0
                        var_theor = sigma_ar1 ** 2 / (1 - phi_ar1 ** 2)
                        st.write(f"Moyenne: {mu_theor}")
                        st.write(f"Variance: {var_theor:.4f}")

                # Simulation
                np.random.seed(42)
                ar1_series = np.zeros(n_ar1)
                epsilon = np.random.normal(0, sigma_ar1, n_ar1)

                for t in range(1, n_ar1):
                    ar1_series[t] = phi_ar1 * ar1_series[t - 1] + epsilon[t]

                with col2:
                    # Graphique de la série
                    fig_ar1 = go.Figure()
                    fig_ar1.add_trace(
                        go.Scatter(y=ar1_series, mode='lines', name='AR(1)', line=dict(color='blue', width=1.5)))
                    fig_ar1.update_layout(
                        title=f'Simulation AR(1) avec φ = {phi_ar1}',
                        xaxis_title='Temps',
                        yaxis_title='Valeur',
                        height=300
                    )
                    st.plotly_chart(fig_ar1, use_container_width=True)

                # ACF et PACF
                st.markdown("### 📊 Fonctions d'Autocorrélation")

                acf_ar1 = acf(ar1_series, nlags=20)
                pacf_ar1 = pacf(ar1_series, nlags=20)

                fig_acf = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])

                # ACF
                fig_acf.add_trace(go.Bar(y=acf_ar1, name='ACF', marker_color='lightblue'), row=1, col=1)
                fig_acf.add_hline(y=1.96 / np.sqrt(n_ar1), line_dash="dash", line_color="red", row=1, col=1)
                fig_acf.add_hline(y=-1.96 / np.sqrt(n_ar1), line_dash="dash", line_color="red", row=1, col=1)

                # PACF
                fig_acf.add_trace(go.Bar(y=pacf_ar1, name='PACF', marker_color='lightcoral'), row=1, col=2)
                fig_acf.add_hline(y=1.96 / np.sqrt(n_ar1), line_dash="dash", line_color="red", row=1, col=2)
                fig_acf.add_hline(y=-1.96 / np.sqrt(n_ar1), line_dash="dash", line_color="red", row=1, col=2)

                fig_acf.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation AR(1)</h4>
                <ul>
                    <li><strong>ACF:</strong> Décroissance exponentielle (géométrique)</li>
                    <li><strong>PACF:</strong> Un seul pic significatif au lag 1, puis 0</li>
                    <li>Ce pattern identifie clairement un processus AR(1)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            elif ordre_ar == "AR(2)":
                st.markdown("### 📐 Processus AR(2)")

                st.latex(r"X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t")

                col1, col2 = st.columns([1, 2])

                with col1:
                    phi1_ar2 = st.slider("Coefficient φ₁:", -1.99, 1.99, 0.5, 0.01, key='phi1_ar2')
                    phi2_ar2 = st.slider("Coefficient φ₂:", -1.0, 1.0, 0.3, 0.01, key='phi2_ar2')
                    sigma_ar2 = st.slider("σ (bruit blanc):", 0.1, 3.0, 1.0, 0.1, key='sigma_ar2')
                    n_ar2 = st.slider("Nombre observations:", 100, 500, 250, key='n_ar2')

                    # Conditions de stationnarité AR(2)
                    cond1 = phi1_ar2 + phi2_ar2 < 1
                    cond2 = phi2_ar2 - phi1_ar2 < 1
                    cond3 = abs(phi2_ar2) < 1

                    if cond1 and cond2 and cond3:
                        st.success("✅ Processus stationnaire")
                    else:
                        st.error("❌ Processus non-stationnaire")

                    st.markdown("**Conditions de stationnarité:**")
                    st.write(f"φ₁ + φ₂ < 1: {'✅' if cond1 else '❌'} ({phi1_ar2 + phi2_ar2:.2f})")
                    st.write(f"φ₂ - φ₁ < 1: {'✅' if cond2 else '❌'} ({phi2_ar2 - phi1_ar2:.2f})")
                    st.write(f"|φ₂| < 1: {'✅' if cond3 else '❌'} ({abs(phi2_ar2):.2f})")

                # Simulation AR(2)
                np.random.seed(42)
                ar2_series = np.zeros(n_ar2)
                epsilon = np.random.normal(0, sigma_ar2, n_ar2)

                for t in range(2, n_ar2):
                    ar2_series[t] = phi1_ar2 * ar2_series[t - 1] + phi2_ar2 * ar2_series[t - 2] + epsilon[t]

                with col2:
                    fig_ar2 = go.Figure()
                    fig_ar2.add_trace(
                        go.Scatter(y=ar2_series, mode='lines', name='AR(2)', line=dict(color='green', width=1.5)))
                    fig_ar2.update_layout(
                        title=f'Simulation AR(2) avec φ₁={phi1_ar2}, φ₂={phi2_ar2}',
                        xaxis_title='Temps',
                        yaxis_title='Valeur',
                        height=300
                    )
                    st.plotly_chart(fig_ar2, use_container_width=True)

                # ACF et PACF
                acf_ar2 = acf(ar2_series, nlags=20)
                pacf_ar2 = pacf(ar2_series, nlags=20)

                fig_acf2 = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])

                fig_acf2.add_trace(go.Bar(y=acf_ar2, name='ACF', marker_color='lightblue'), row=1, col=1)
                fig_acf2.add_hline(y=1.96 / np.sqrt(n_ar2), line_dash="dash", line_color="red", row=1, col=1)
                fig_acf2.add_hline(y=-1.96 / np.sqrt(n_ar2), line_dash="dash", line_color="red", row=1, col=1)

                fig_acf2.add_trace(go.Bar(y=pacf_ar2, name='PACF', marker_color='lightcoral'), row=1, col=2)
                fig_acf2.add_hline(y=1.96 / np.sqrt(n_ar2), line_dash="dash", line_color="red", row=1, col=2)
                fig_acf2.add_hline(y=-1.96 / np.sqrt(n_ar2), line_dash="dash", line_color="red", row=1, col=2)

                fig_acf2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf2, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation AR(2)</h4>
                <ul>
                    <li><strong>ACF:</strong> Décroissance (peut être oscillante si racines complexes)</li>
                    <li><strong>PACF:</strong> Deux pics significatifs (lags 1 et 2), puis 0</li>
                    <li>Identification: PACF tronquée au lag p pour AR(p)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:  # AR(3)
                st.markdown("### 📐 Processus AR(3)")

                st.latex(r"X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \phi_3 X_{t-3} + \epsilon_t")

                col1, col2 = st.columns([1, 2])

                with col1:
                    phi1_ar3 = st.slider("φ₁:", -1.0, 1.0, 0.4, 0.01, key='phi1_ar3')
                    phi2_ar3 = st.slider("φ₂:", -1.0, 1.0, 0.3, 0.01, key='phi2_ar3')
                    phi3_ar3 = st.slider("φ₃:", -1.0, 1.0, 0.2, 0.01, key='phi3_ar3')
                    sigma_ar3 = st.slider("σ:", 0.1, 3.0, 1.0, 0.1, key='sigma_ar3')
                    n_ar3 = st.slider("n:", 100, 500, 250, key='n_ar3')

                np.random.seed(42)
                ar3_series = np.zeros(n_ar3)
                epsilon = np.random.normal(0, sigma_ar3, n_ar3)

                for t in range(3, n_ar3):
                    ar3_series[t] = (phi1_ar3 * ar3_series[t - 1] +
                                     phi2_ar3 * ar3_series[t - 2] +
                                     phi3_ar3 * ar3_series[t - 3] +
                                     epsilon[t])

                with col2:
                    fig_ar3 = go.Figure()
                    fig_ar3.add_trace(go.Scatter(y=ar3_series, mode='lines', line=dict(color='purple', width=1.5)))
                    fig_ar3.update_layout(
                        title=f'AR(3): φ₁={phi1_ar3}, φ₂={phi2_ar3}, φ₃={phi3_ar3}',
                        height=300
                    )
                    st.plotly_chart(fig_ar3, use_container_width=True)

                # ACF et PACF
                acf_ar3 = acf(ar3_series, nlags=20)
                pacf_ar3 = pacf(ar3_series, nlags=20)

                fig_acf3 = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
                fig_acf3.add_trace(go.Bar(y=acf_ar3, marker_color='lightblue'), row=1, col=1)
                fig_acf3.add_trace(go.Bar(y=pacf_ar3, marker_color='lightcoral'), row=1, col=2)

                fig_acf3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf3, use_container_width=True)

        with tab3:
            st.markdown("### 📈 Processus Moyenne Mobile MA(q)")

            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Définition</h3>
            <p>Un processus MA(q) est une combinaison linéaire des q dernières innovations (erreurs).</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(
                r"X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}")

            ordre_ma = st.selectbox("Choisissez l'ordre:", ["MA(1)", "MA(2)", "MA(3)"])

            if ordre_ma == "MA(1)":
                st.markdown("### 📐 Processus MA(1)")

                st.latex(r"X_t = \epsilon_t + \theta \epsilon_{t-1}")

                col1, col2 = st.columns([1, 2])

                with col1:
                    theta_ma1 = st.slider("Coefficient θ:", -0.99, 0.99, 0.6, 0.01, key='theta_ma1')
                    sigma_ma1 = st.slider("σ:", 0.1, 3.0, 1.0, 0.1, key='sigma_ma1')
                    n_ma1 = st.slider("n:", 100, 500, 250, key='n_ma1')

                    if abs(theta_ma1) < 1:
                        st.success(f"✅ Processus inversible (|θ| = {abs(theta_ma1):.2f} < 1)")
                    else:
                        st.warning(f"⚠️ Processus non-inversible (|θ| = {abs(theta_ma1):.2f} ≥ 1)")

                    st.markdown("**Propriétés théoriques:**")
                    st.write("Moyenne: 0")
                    var_ma1 = sigma_ma1 ** 2 * (1 + theta_ma1 ** 2)
                    st.write(f"Variance: {var_ma1:.4f}")

                # Simulation MA(1)
                np.random.seed(42)
                epsilon_ma1 = np.random.normal(0, sigma_ma1, n_ma1)
                ma1_series = np.zeros(n_ma1)
                ma1_series[0] = epsilon_ma1[0]

                for t in range(1, n_ma1):
                    ma1_series[t] = epsilon_ma1[t] + theta_ma1 * epsilon_ma1[t - 1]

                with col2:
                    fig_ma1 = go.Figure()
                    fig_ma1.add_trace(go.Scatter(y=ma1_series, mode='lines', line=dict(color='red', width=1.5)))
                    fig_ma1.update_layout(
                        title=f'MA(1) avec θ = {theta_ma1}',
                        xaxis_title='Temps',
                        yaxis_title='Valeur',
                        height=300
                    )
                    st.plotly_chart(fig_ma1, use_container_width=True)

                # ACF et PACF
                acf_ma1 = acf(ma1_series, nlags=20)
                pacf_ma1 = pacf(ma1_series, nlags=20)

                fig_acf_ma = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])

                fig_acf_ma.add_trace(go.Bar(y=acf_ma1, marker_color='lightblue'), row=1, col=1)
                fig_acf_ma.add_hline(y=1.96 / np.sqrt(n_ma1), line_dash="dash", line_color="red", row=1, col=1)
                fig_acf_ma.add_hline(y=-1.96 / np.sqrt(n_ma1), line_dash="dash", line_color="red", row=1, col=1)

                fig_acf_ma.add_trace(go.Bar(y=pacf_ma1, marker_color='lightcoral'), row=1, col=2)
                fig_acf_ma.add_hline(y=1.96 / np.sqrt(n_ma1), line_dash="dash", line_color="red", row=1, col=2)
                fig_acf_ma.add_hline(y=-1.96 / np.sqrt(n_ma1), line_dash="dash", line_color="red", row=1, col=2)

                fig_acf_ma.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf_ma, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation MA(1)</h4>
                <ul>
                    <li><strong>ACF:</strong> Un seul pic significatif au lag 1, puis 0</li>
                    <li><strong>PACF:</strong> Décroissance exponentielle (géométrique)</li>
                    <li>Pattern inverse de AR(1)!</li>
                    <li>Mémoire courte: influence seulement sur 1 période</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                # ACF théorique
                st.markdown("### 📊 ACF Théorique vs Empirique")

                rho_1_theor = theta_ma1 / (1 + theta_ma1 ** 2)

                st.markdown(f"""
                <div class="formula-box">
                <h4>ACF Théorique MA(1)</h4>
                <p>ρ(1) = θ / (1 + θ²) = {rho_1_theor:.4f}</p>
                <p>ρ(h) = 0 pour h > 1</p>
                <p><strong>ACF empirique au lag 1:</strong> {acf_ma1[1]:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

            elif ordre_ma == "MA(2)":
                st.markdown("### 📐 Processus MA(2)")

                st.latex(r"X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}")

                col1, col2 = st.columns([1, 2])

                with col1:
                    theta1_ma2 = st.slider("θ₁:", -0.99, 0.99, 0.5, 0.01, key='theta1_ma2')
                    theta2_ma2 = st.slider("θ₂:", -0.99, 0.99, 0.3, 0.01, key='theta2_ma2')
                    sigma_ma2 = st.slider("σ:", 0.1, 3.0, 1.0, 0.1, key='sigma_ma2')
                    n_ma2 = st.slider("n:", 100, 500, 250, key='n_ma2')

                # Simulation MA(2)
                np.random.seed(42)
                epsilon_ma2 = np.random.normal(0, sigma_ma2, n_ma2)
                ma2_series = np.zeros(n_ma2)

                for t in range(2, n_ma2):
                    ma2_series[t] = (epsilon_ma2[t] +
                                     theta1_ma2 * epsilon_ma2[t - 1] +
                                     theta2_ma2 * epsilon_ma2[t - 2])

                with col2:
                    fig_ma2 = go.Figure()
                    fig_ma2.add_trace(go.Scatter(y=ma2_series, mode='lines', line=dict(color='orange', width=1.5)))
                    fig_ma2.update_layout(
                        title=f'MA(2): θ₁={theta1_ma2}, θ₂={theta2_ma2}',
                        height=300
                    )
                    st.plotly_chart(fig_ma2, use_container_width=True)

                # ACF et PACF
                acf_ma2 = acf(ma2_series, nlags=20)
                pacf_ma2 = pacf(ma2_series, nlags=20)

                fig_acf_ma2 = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
                fig_acf_ma2.add_trace(go.Bar(y=acf_ma2, marker_color='lightblue'), row=1, col=1)
                fig_acf_ma2.add_trace(go.Bar(y=pacf_ma2, marker_color='lightcoral'), row=1, col=2)
                fig_acf_ma2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf_ma2, use_container_width=True)

                st.markdown("""
                <div class="example-box">
                <h4>💡 Interprétation MA(2)</h4>
                <ul>
                    <li><strong>ACF:</strong> Deux pics significatifs (lags 1 et 2), puis 0</li>
                    <li><strong>PACF:</strong> Décroissance exponentielle</li>
                    <li>ACF tronquée au lag q pour MA(q)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:  # MA(3)
                st.markdown("### 📐 Processus MA(3)")

                st.latex(
                    r"X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \theta_3 \epsilon_{t-3}")

                col1, col2 = st.columns([1, 2])

                with col1:
                    theta1_ma3 = st.slider("θ₁:", -0.99, 0.99, 0.4, 0.01)
                    theta2_ma3 = st.slider("θ₂:", -0.99, 0.99, 0.3, 0.01)
                    theta3_ma3 = st.slider("θ₃:", -0.99, 0.99, 0.2, 0.01)
                    sigma_ma3 = st.slider("σ:", 0.1, 3.0, 1.0, 0.1)
                    n_ma3 = st.slider("n:", 100, 500, 250)

                np.random.seed(42)
                epsilon_ma3 = np.random.normal(0, sigma_ma3, n_ma3)
                ma3_series = np.zeros(n_ma3)

                for t in range(3, n_ma3):
                    ma3_series[t] = (epsilon_ma3[t] +
                                     theta1_ma3 * epsilon_ma3[t - 1] +
                                     theta2_ma3 * epsilon_ma3[t - 2] +
                                     theta3_ma3 * epsilon_ma3[t - 3])

                with col2:
                    fig_ma3 = go.Figure()
                    fig_ma3.add_trace(go.Scatter(y=ma3_series, mode='lines', line=dict(color='darkred', width=1.5)))
                    fig_ma3.update_layout(title='MA(3)', height=300)
                    st.plotly_chart(fig_ma3, use_container_width=True)

                acf_ma3 = acf(ma3_series, nlags=20)
                pacf_ma3 = pacf(ma3_series, nlags=20)

                fig_acf_ma3 = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])
                fig_acf_ma3.add_trace(go.Bar(y=acf_ma3, marker_color='lightblue'), row=1, col=1)
                fig_acf_ma3.add_trace(go.Bar(y=pacf_ma3, marker_color='lightcoral'), row=1, col=2)
                fig_acf_ma3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_acf_ma3, use_container_width=True)

        with tab4:
            st.markdown("### 🔄 Processus ARMA(p,q)")

            st.markdown("""
            <div class="theory-box">
            <h3>🎯 Combinaison AR + MA</h3>
            <p>ARMA(p,q) combine les avantages des deux approches pour une modélisation optimale.</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}")

            st.markdown("**Forme opérateur:**")
            st.latex(r"\Phi(L) X_t = \Theta(L) \epsilon_t")

            st.markdown("### 🎛️ Simulateur ARMA Interactif")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Partie AR**")
                p_arma = st.selectbox("Ordre p:", [0, 1, 2, 3], index=1)

                phi_arma = []
                for i in range(p_arma):
                    phi_arma.append(st.slider(f"φ_{i + 1}:", -0.99, 0.99, 0.5 / (i + 1), 0.01, key=f'phi_arma_{i}'))

            with col2:
                st.markdown("**Partie MA**")
                q_arma = st.selectbox("Ordre q:", [0, 1, 2, 3], index=1)

                theta_arma = []
                for j in range(q_arma):
                    theta_arma.append(st.slider(f"θ_{j + 1}:", -0.99, 0.99, 0.3 / (j + 1), 0.01, key=f'theta_arma_{j}'))

            with col3:
                st.markdown("**Paramètres**")
                sigma_arma = st.slider("σ:", 0.1, 3.0, 1.0, 0.1, key='sigma_arma')
                n_arma = st.slider("n:", 100, 1000, 500, key='n_arma')

            # Simulation ARMA
            if p_arma > 0 or q_arma > 0:
                try:
                    ar_params = np.array([1] + [-phi for phi in phi_arma])
                    ma_params = np.array([1] + theta_arma)

                    arma_process = ArmaProcess(ar_params, ma_params)

                    if arma_process.isstationary and arma_process.isinvertible:
                        arma_series = arma_process.generate_sample(n_arma, scale=sigma_arma)

                        # Graphique
                        fig_arma = go.Figure()
                        fig_arma.add_trace(go.Scatter(
                            y=arma_series,
                            mode='lines',
                            name=f'ARMA({p_arma},{q_arma})',
                            line=dict(color='darkblue', width=1.5)
                        ))
                        fig_arma.update_layout(
                            title=f'Simulation ARMA({p_arma},{q_arma})',
                            xaxis_title='Temps',
                            yaxis_title='Valeur',
                            height=400
                        )
                        st.plotly_chart(fig_arma, use_container_width=True)

                        # ACF et PACF
                        acf_arma = acf(arma_series, nlags=30)
                        pacf_arma = pacf(arma_series, nlags=30)

                        fig_acf_arma = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])

                        fig_acf_arma.add_trace(go.Bar(y=acf_arma, marker_color='lightblue'), row=1, col=1)
                        fig_acf_arma.add_hline(y=1.96 / np.sqrt(n_arma), line_dash="dash", line_color="red", row=1,
                                               col=1)
                        fig_acf_arma.add_hline(y=-1.96 / np.sqrt(n_arma), line_dash="dash", line_color="red", row=1,
                                               col=1)

                        fig_acf_arma.add_trace(go.Bar(y=pacf_arma, marker_color='lightcoral'), row=1, col=2)
                        fig_acf_arma.add_hline(y=1.96 / np.sqrt(n_arma), line_dash="dash", line_color="red", row=1,
                                               col=2)
                        fig_acf_arma.add_hline(y=-1.96 / np.sqrt(n_arma), line_dash="dash", line_color="red", row=1,
                                               col=2)

                        fig_acf_arma.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_acf_arma, use_container_width=True)

                        st.success("✅ Processus stationnaire et inversible")

                        # Statistiques
                        st.markdown("### 📊 Statistiques")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        col_stat1.metric("Moyenne", f"{np.mean(arma_series):.4f}")
                        col_stat2.metric("Écart-type", f"{np.std(arma_series):.4f}")
                        col_stat3.metric("Min", f"{np.min(arma_series):.4f}")
                        col_stat4.metric("Max", f"{np.max(arma_series):.4f}")

                    else:
                        if not arma_process.isstationary:
                            st.error("❌ Processus non-stationnaire! Ajustez les paramètres AR.")
                        if not arma_process.isinvertible:
                            st.warning("⚠️ Processus non-inversible! Ajustez les paramètres MA.")

                except Exception as e:
                    st.error(f"Erreur de simulation: {str(e)}")
            else:
                st.info("Choisissez au moins p>0 ou q>0 pour simuler un processus ARMA")

            st.markdown("---")

            st.markdown("""
            <div class="theory-box">
            <h3>📋 Guide d'Identification ARMA</h3>
            </div>
            """, unsafe_allow_html=True)

            identification_df = pd.DataFrame({
                'Modèle': ['AR(p)', 'MA(q)', 'ARMA(p,q)'],
                'ACF': ['Décroît exponentiellement', 'Tronquée au lag q', 'Décroît exponentiellement'],
                'PACF': ['Tronquée au lag p', 'Décroît exponentiellement', 'Décroît exponentiellement']
            })

            st.table(identification_df)

        with tab5:
            st.markdown("### 🎯 Application Pratique: Estimation d'un Modèle ARMA")

            st.markdown("""
            <div class="example-box">
            <h3>📊 Étude de Cas Complète</h3>
            <p>Nous allons suivre toutes les étapes pour modéliser une série temporelle.</p>
            </div>
            """, unsafe_allow_html=True)

            # Choix: données simulées ou exemple
            data_choice = st.radio(
                "Source des données:",
                ["Simuler un processus ARMA", "Utiliser des données d'exemple (prix quotidiens)"]
            )

            if data_choice == "Simuler un processus ARMA":
                st.markdown("### 1️⃣ Générer les données")

                col_gen1, col_gen2 = st.columns(2)

                with col_gen1:
                    true_p = st.selectbox("Ordre AR (vrai):", [0, 1, 2], index=1, key='true_p')
                    true_phi = []
                    for i in range(true_p):
                        true_phi.append(
                            st.slider(f"φ_{i + 1} (vrai):", -0.9, 0.9, 0.6 / (i + 1), 0.1, key=f'true_phi_{i}'))

                with col_gen2:
                    true_q = st.selectbox("Ordre MA (vrai):", [0, 1, 2], index=1, key='true_q')
                    true_theta = []
                    for j in range(true_q):
                        true_theta.append(
                            st.slider(f"θ_{j + 1} (vrai):", -0.9, 0.9, 0.4 / (j + 1), 0.1, key=f'true_theta_{j}'))

                n_sample = st.slider("Taille de l'échantillon:", 100, 1000, 300)

                # Génération
                ar_true = np.array([1] + [-phi for phi in true_phi])
                ma_true = np.array([1] + true_theta)

                arma_true = ArmaProcess(ar_true, ma_true)
                data_series = arma_true.generate_sample(n_sample, scale=1.0)

                st.success(f"✅ Données générées: ARMA({true_p},{true_q}) avec {n_sample} observations")

            else:
                # Simuler des "prix quotidiens" réalistes
                st.markdown("### 1️⃣ Chargement des données")

                np.random.seed(123)
                n_sample = 500

                # Prix (log-rendements ARMA)
                returns_arma = ArmaProcess([1, -0.3], [1, 0.5]).generate_sample(n_sample, scale=0.01)
                prix = 100 * np.exp(np.cumsum(returns_arma))

                # On travaille avec les rendements
                data_series = np.diff(np.log(prix))

                st.info("📊 Données: Rendements logarithmiques quotidiens (n=499)")

            # Visualisation
            fig_data = go.Figure()
            fig_data.add_trace(go.Scatter(
                y=data_series,
                mode='lines',
                name='Série observée',
                line=dict(color='black', width=1)
            ))
            fig_data.update_layout(
                title='Série Temporelle Observée',
                xaxis_title='Temps',
                yaxis_title='Valeur',
                height=300
            )
            st.plotly_chart(fig_data, use_container_width=True)

            # Étape 2: ACF/PACF
            st.markdown("### 2️⃣ Analyse ACF/PACF pour identifier l'ordre")

            nlags_diag = st.slider("Nombre de lags à afficher:", 10, 50, 30)

            acf_data = acf(data_series, nlags=nlags_diag)
            pacf_data = pacf(data_series, nlags=nlags_diag)

            fig_diag = make_subplots(rows=1, cols=2, subplot_titles=['ACF', 'PACF'])

            fig_diag.add_trace(go.Bar(y=acf_data, marker_color='steelblue'), row=1, col=1)
            fig_diag.add_hline(y=1.96 / np.sqrt(len(data_series)), line_dash="dash", line_color="red", row=1, col=1)
            fig_diag.add_hline(y=-1.96 / np.sqrt(len(data_series)), line_dash="dash", line_color="red", row=1, col=1)

            fig_diag.add_trace(go.Bar(y=pacf_data, marker_color='coral'), row=1, col=2)
            fig_diag.add_hline(y=1.96 / np.sqrt(len(data_series)), line_dash="dash", line_color="red", row=1, col=2)
            fig_diag.add_hline(y=-1.96 / np.sqrt(len(data_series)), line_dash="dash", line_color="red", row=1, col=2)

            fig_diag.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_diag, use_container_width=True)

            # Étape 3: Estimation
            st.markdown("### 3️⃣ Estimation du modèle ARMA")

            col_est1, col_est2 = st.columns(2)

            with col_est1:
                p_est = st.selectbox("Ordre AR à estimer:", [0, 1, 2, 3], index=1, key='p_est')

            with col_est2:
                q_est = st.selectbox("Ordre MA à estimer:", [0, 1, 2, 3], index=1, key='q_est')

            if st.button("🚀 Estimer le modèle ARMA"):
                try:
                    model_arma = ARIMA(data_series, order=(p_est, 0, q_est))
                    results_arma = model_arma.fit()

                    st.success(f"✅ Modèle ARMA({p_est},{q_est}) estimé avec succès!")

                    # Résumé
                    st.markdown("### 📊 Résultats de l'estimation")

                    st.text(results_arma.summary())

                    # Coefficients
                    st.markdown("### 📈 Coefficients estimés")

                    coef_df = pd.DataFrame({
                        'Paramètre': results_arma.params.index,
                        'Valeur': results_arma.params.values,
                        'Std Error': results_arma.bse.values,
                        'P-value': results_arma.pvalues.values
                    })

                    st.dataframe(coef_df.style.format({'Valeur': '{:.4f}', 'Std Error': '{:.4f}', 'P-value': '{:.4f}'}))

                    # Critères d'information
                    st.markdown("### 📏 Critères d'Information")

                    col_aic, col_bic = st.columns(2)
                    col_aic.metric("AIC", f"{results_arma.aic:.2f}")
                    col_bic.metric("BIC", f"{results_arma.bic:.2f}")

                    # Diagnostic des résidus
                    st.markdown("### 4️⃣ Diagnostic des Résidus")

                    residus = results_arma.resid

                    fig_resid = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Résidus', 'Histogramme', 'ACF des résidus', 'Q-Q Plot']
                    )

                    # Résidus
                    fig_resid.add_trace(go.Scatter(y=residus, mode='lines', line=dict(color='green')), row=1, col=1)

                    # Histogramme
                    fig_resid.add_trace(go.Histogram(x=residus, nbinsx=30, marker_color='lightgreen'), row=1, col=2)

                    # ACF résidus
                    acf_resid = acf(residus, nlags=20)
                    fig_resid.add_trace(go.Bar(y=acf_resid, marker_color='lightblue'), row=2, col=1)

                    # Q-Q plot
                    from scipy import stats as sp_stats

                    qq = sp_stats.probplot(residus, dist="norm")
                    fig_resid.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='purple')),
                                        row=2, col=2)
                    fig_resid.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                                                   mode='lines', line=dict(color='red', dash='dash')), row=2, col=2)

                    fig_resid.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_resid, use_container_width=True)

                    # Test de Ljung-Box
                    st.markdown("### 🔍 Test de Ljung-Box (Bruit Blanc)")

                    lb_test = acorr_ljungbox(residus, lags=[10, 20], return_df=True)
                    st.dataframe(lb_test)

                    if lb_test['lb_pvalue'].min() > 0.05:
                        st.success("✅ Les résidus ressemblent à un bruit blanc (p > 0.05)")
                    else:
                        st.warning("⚠️ Les résidus ne sont pas un bruit blanc (p < 0.05). Essayez un autre modèle.")

                    # Prévisions
                    st.markdown("### 5️⃣ Prévisions")

                    n_forecast = st.slider("Horizon de prévision:", 1, 50, 10)

                    forecast_result = results_arma.get_forecast(steps=n_forecast)
                    forecast_mean = forecast_result.predicted_mean
                    forecast_ci = forecast_result.conf_int()

                    # Graphique prévisions
                    fig_forecast = go.Figure()

                    # Données historiques
                    fig_forecast.add_trace(go.Scatter(
                        y=data_series[-100:],
                        mode='lines',
                        name='Données historiques',
                        line=dict(color='black')
                    ))

                    # Prévisions
                    forecast_index = np.arange(len(data_series), len(data_series) + n_forecast)

                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_index,
                        y=forecast_mean,
                        mode='lines',
                        name='Prévision',
                        line=dict(color='red', dash='dash')
                    ))

                    # Intervalle de confiance
                    fig_forecast.add_trace(go.Scatter(
                        x=np.concatenate([forecast_index, forecast_index[::-1]]),
                        y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,0,0,0)'),
                        name='IC 95%'
                    ))

                    fig_forecast.update_layout(
                        title=f'Prévisions ARMA({p_est},{q_est}) - Horizon {n_forecast}',
                        xaxis_title='Temps',
                        yaxis_title='Valeur',
                        height=400
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur lors de l'estimation: {str(e)}")

            st.markdown("""
            <div class="theory-box">
            <h3>📚 Résumé de la Méthodologie Box-Jenkins</h3>
            <ol>
                <li><strong>Identification:</strong> ACF/PACF pour choisir p et q</li>
                <li><strong>Estimation:</strong> Maximum de vraisemblance</li>
                <li><strong>Validation:</strong> Diagnostic des résidus (bruit blanc?)</li>
                <li><strong>Prévision:</strong> Utiliser le modèle validé</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

# ==================== CHAPITRE 2 ====================
elif chapitre == "🔍 Chapitre 2: Tests de Stationnarité":

    st.sidebar.markdown("---")
    section_ch2 = st.sidebar.selectbox(
        "Choisissez une section:",
        ["2.1 - Processus non stationnaires",
         "2.2 - Test de Dickey-Fuller",
         "2.3 - Test ADF (Augmented Dickey-Fuller)",
         "2.4 - Processus ARIMA"]
    )

    # ========== SECTION 2.1 ==========
    if section_ch2 == "2.1 - Processus non stationnaires":
        st.markdown('<p class="sub-header">2.1 - Processus Non Stationnaires</p>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📚 Théorie", "🔍 Types de Non-Stationnarité", "🎨 Visualisations"])

        with tab1:
            st.markdown("""
            <div class="warning-box">
            <h3>⚠️ Qu'est-ce qu'un Processus Non-Stationnaire?</h3>
            <p>Un processus est <strong>non-stationnaire</strong> si au moins une de ses propriétés statistiques
            <strong>change dans le temps</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔑 Caractéristiques des Processus Non-Stationnaires")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>📊 Processus Stationnaire</h4>
                <ul>
                    <li>Moyenne constante: E[Xₜ] = μ</li>
                    <li>Variance constante: Var[Xₜ] = σ²</li>
                    <li>Autocovariance dépend seulement du décalage</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Processus Non-Stationnaire</h4>
                <ul>
                    <li>Moyenne variable: E[Xₜ] ≠ constante</li>
                    <li>Variance variable: Var[Xₜ] ≠ constante</li>
                    <li>Tendance ou saisonnalité présente</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📐 Définition Mathématique")

            st.latex(r"""
            \text{Un processus } X_t \text{ est stationnaire si:}
            """)

            st.latex(r"""
            \begin{cases}
            E[X_t] = \mu & \forall t \\
            Var[X_t] = \sigma^2 & \forall t \\
            Cov[X_t, X_{t+h}] = \gamma(h) & \forall t, h
            \end{cases}
            """)

            st.markdown("""
            <div class="formula-box">
            <p><strong>Si au moins une condition n'est pas satisfaite</strong> → Processus NON-STATIONNAIRE</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 🎯 Types de Non-Stationnarité")

            type_ns = st.selectbox(
                "Sélectionnez le type à explorer:",
                ["Tendance Déterministe (TS)",
                 "Tendance Stochastique (DS)",
                 "Variance Non-Constante",
                 "Comparaison des Types"]
            )

            if type_ns == "Tendance Déterministe (TS)":
                st.markdown("""
                <div class="info-box">
                <h4>📈 Trend-Stationary (TS)</h4>
                <p>Le processus contient une <strong>tendance déterministe</strong> qui peut être modélisée par une fonction du temps.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                X_t = \alpha + \beta t + \varepsilon_t
                """)

                st.markdown("""
                Où:
                - α : constante
                - β : coefficient de tendance
                - t : temps
                - εₜ : bruit blanc stationnaire
                """)

                st.markdown("### 🔧 Correction:")
                st.markdown("""
                <div class="success-box">
                <p><strong>Détendanciation (Detrending)</strong>: Soustraire la tendance estimée</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                Y_t = X_t - (\hat{\alpha} + \hat{\beta}t)
                """)

            elif type_ns == "Tendance Stochastique (DS)":
                st.markdown("""
                <div class="warning-box">
                <h4>🎲 Difference-Stationary (DS)</h4>
                <p>Le processus contient une <strong>racine unitaire</strong>.
                La non-stationnarité est de nature <strong>stochastique</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                X_t = X_{t-1} + \varepsilon_t \quad \text{(Marche Aléatoire)}
                """)

                st.markdown("### 📊 Modèle Général avec Dérive:")
                st.latex(r"""
                X_t = \delta + X_{t-1} + \varepsilon_t
                """)

                st.markdown("### 🔧 Correction:")
                st.markdown("""
                <div class="success-box">
                <p><strong>Différenciation</strong>: Prendre la différence première</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = X_t - X_{t-1} = \delta + \varepsilon_t
                """)

            elif type_ns == "Variance Non-Constante":
                st.markdown("""
                <div class="info-box">
                <h4>📊 Hétéroscédasticité</h4>
                <p>La variance du processus <strong>change dans le temps</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                Var[X_t] = \sigma_t^2 \neq \sigma^2
                """)

                st.markdown("### 🔧 Correction:")
                st.markdown("""
                <div class="success-box">
                <p><strong>Transformation</strong>:</p>
                <ul>
                    <li>Logarithme: log(Xₜ)</li>
                    <li>Racine carrée: √Xₜ</li>
                    <li>Transformation de Box-Cox</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:  # Comparaison
                st.markdown("### 📊 Tableau Comparatif")

                comparison_df = pd.DataFrame({
                    'Caractéristique': ['Nature', 'Modèle', 'Correction', 'Exemple', 'Réversion'],
                    'TS (Trend-Stationary)': [
                        'Déterministe',
                        'Xₜ = α + βt + εₜ',
                        'Détendanciation',
                        'PIB avec tendance linéaire',
                        'Revient à la tendance'
                    ],
                    'DS (Difference-Stationary)': [
                        'Stochastique',
                        'Xₜ = Xₜ₋₁ + εₜ',
                        'Différenciation',
                        'Prix boursiers',
                        'Pas de réversion'
                    ]
                })

                st.dataframe(comparison_df, use_container_width=True)

        with tab3:
            st.markdown("### 🎨 Visualisations Interactives")

            st.markdown("#### Paramètres de Simulation")
            col1, col2, col3 = st.columns(3)

            with col1:
                n_points = st.slider("Nombre de points", 100, 500, 200)
            with col2:
                trend_coef = st.slider("Coefficient de tendance (β)", 0.0, 2.0, 0.5)
            with col3:
                noise_std = st.slider("Écart-type du bruit", 0.1, 5.0, 1.0)

            # Simulation
            np.random.seed(42)
            t = np.arange(n_points)
            epsilon = np.random.normal(0, noise_std, n_points)

            # Processus TS
            ts_process = 10 + trend_coef * t + epsilon

            # Processus DS (Random Walk)
            ds_process = np.cumsum(epsilon) + 10

            # Processus Stationnaire
            stationary = epsilon + 10

            # Graphiques
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Processus TS', 'ACF - TS',
                                'Processus DS', 'ACF - DS',
                                'Processus Stationnaire', 'ACF - Stationnaire'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # TS
            fig.add_trace(go.Scatter(x=t, y=ts_process, mode='lines',
                                     name='TS', line=dict(color='red')), row=1, col=1)

            # ACF TS
            acf_ts = [np.corrcoef(ts_process[:-i], ts_process[i:])[0, 1] if i > 0 else 1
                      for i in range(min(40, n_points // 4))]
            fig.add_trace(go.Bar(x=list(range(len(acf_ts))), y=acf_ts,
                                 name='ACF-TS', marker_color='red'), row=1, col=2)

            # DS
            fig.add_trace(go.Scatter(x=t, y=ds_process, mode='lines',
                                     name='DS', line=dict(color='blue')), row=2, col=1)

            # ACF DS
            acf_ds = [np.corrcoef(ds_process[:-i], ds_process[i:])[0, 1] if i > 0 else 1
                      for i in range(min(40, n_points // 4))]
            fig.add_trace(go.Bar(x=list(range(len(acf_ds))), y=acf_ds,
                                 name='ACF-DS', marker_color='blue'), row=2, col=2)

            # Stationnaire
            fig.add_trace(go.Scatter(x=t, y=stationary, mode='lines',
                                     name='Stationnaire', line=dict(color='green')), row=3, col=1)

            # ACF Stationnaire
            acf_stat = [np.corrcoef(stationary[:-i], stationary[i:])[0, 1] if i > 0 else 1
                        for i in range(min(40, n_points // 4))]
            fig.add_trace(go.Bar(x=list(range(len(acf_stat))), y=acf_stat,
                                 name='ACF-Stat', marker_color='green'), row=3, col=2)

            fig.update_layout(height=900, showlegend=False, title_text="Comparaison des Processus")
            st.plotly_chart(fig, use_container_width=True)

            # Statistiques
            st.markdown("### 📊 Statistiques Descriptives")
            stats_df = pd.DataFrame({
                'Processus': ['TS', 'DS', 'Stationnaire'],
                'Moyenne': [np.mean(ts_process), np.mean(ds_process), np.mean(stationary)],
                'Écart-type': [np.std(ts_process), np.std(ds_process), np.std(stationary)],
                'Min': [np.min(ts_process), np.min(ds_process), np.min(stationary)],
                'Max': [np.max(ts_process), np.max(ds_process), np.max(stationary)]
            })
            st.dataframe(stats_df.style.format({'Moyenne': '{:.2f}',
                                                'Écart-type': '{:.2f}',
                                                'Min': '{:.2f}',
                                                'Max': '{:.2f}'}), use_container_width=True)

    # ========== SECTION 2.2 ==========
    elif section_ch2 == "2.2 - Test de Dickey-Fuller":
        st.markdown('<p class="sub-header">2.2 - Test de Dickey-Fuller (DF)</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📚 Théorie", "🔢 Modèles DF", "💻 Application", "🎯 Interprétation"])

        with tab1:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Objectif du Test de Dickey-Fuller</h3>
            <p>Tester la présence d'une <strong>racine unitaire</strong> dans une série temporelle.</p>
            <p>C'est-à-dire: déterminer si la série est <strong>stationnaire</strong> ou <strong>non-stationnaire</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📖 Contexte Historique")
            st.markdown("""
            - **Développé par**: David Dickey et Wayne Fuller (1979)
            - **Application**: Économétrie, finance, analyse de séries temporelles
            - **Importance**: Test fondamental avant toute modélisation ARIMA
            """)

            st.markdown("### 🔬 Principe du Test")

            st.markdown("""
            <div class="formula-box">
            <h4>Modèle AR(1) général:</h4>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            X_t = \phi X_{t-1} + \varepsilon_t
            """)

            st.markdown("**Trois cas possibles:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="success-box">
                <h4>|φ| < 1</h4>
                <p>✅ Stationnaire</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="warning-box">
                <h4>φ = 1</h4>
                <p>⚠️ Racine unitaire<br>(Non-stationnaire)</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="error-box">
                <h4>|φ| > 1</h4>
                <p>❌ Explosif</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🎯 Hypothèses du Test")

            st.markdown("""
            <div class="info-box">
            <h4>Test de racine unitaire:</h4>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            \begin{cases}
            H_0: \phi = 1 & \text{(Racine unitaire - Non-stationnaire)} \\
            H_1: |\phi| < 1 & \text{(Stationnaire)}
            \end{cases}
            """)

            st.markdown("""
            <div class="warning-box">
            <p><strong>⚠️ Important:</strong> Le test DF utilise une distribution particulière
            (distribution de Dickey-Fuller) et NON la distribution de Student classique!</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 🔢 Les Trois Modèles de Dickey-Fuller")

            model_type = st.radio(
                "Sélectionnez le modèle à étudier:",
                ["Modèle 1: Sans constante ni tendance",
                 "Modèle 2: Avec constante",
                 "Modèle 3: Avec constante et tendance"],
                horizontal=False
            )

            if model_type == "Modèle 1: Sans constante ni tendance":
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle 1: Forme la plus simple</h4>
                <p>Utilisé quand la série oscille autour de zéro sans tendance.</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Forme AR(1):**")
                    st.latex(r"""
                    X_t = \phi X_{t-1} + \varepsilon_t
                    """)

                with col2:
                    st.markdown("**Forme DF (différence):**")
                    st.latex(r"""
                    \Delta X_t = \gamma X_{t-1} + \varepsilon_t
                    """)

                st.markdown("**Relation:**")
                st.latex(r"""
                \gamma = \phi - 1
                """)

                st.markdown("**Hypothèses:**")
                st.latex(r"""
                \begin{cases}
                H_0: \gamma = 0 & \text{(Racine unitaire)} \\
                H_1: \gamma < 0 & \text{(Stationnaire)}
                \end{cases}
                """)

                st.markdown("**Statistique de test:**")
                st.latex(r"""
                DF = \frac{\hat{\gamma}}{SE(\hat{\gamma})}
                """)

            elif model_type == "Modèle 2: Avec constante":
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle 2: Avec constante (drift)</h4>
                <p>Utilisé pour les séries avec une moyenne non nulle.</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Forme AR(1):**")
                    st.latex(r"""
                    X_t = c + \phi X_{t-1} + \varepsilon_t
                    """)

                with col2:
                    st.markdown("**Forme DF:**")
                    st.latex(r"""
                    \Delta X_t = c + \gamma X_{t-1} + \varepsilon_t
                    """)

                st.markdown("**Interprétation de c:**")
                st.markdown("""
                - Si γ < 0 et c ≠ 0 : la série est stationnaire autour d'une moyenne c/(1-φ)
                - Si γ = 0 et c ≠ 0 : la série est une marche aléatoire avec dérive
                """)

                st.markdown("**Hypothèses:**")
                st.latex(r"""
                \begin{cases}
                H_0: \gamma = 0 & \text{(Racine unitaire avec dérive)} \\
                H_1: \gamma < 0 & \text{(Stationnaire autour d'une moyenne)}
                \end{cases}
                """)

            else:  # Modèle 3
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle 3: Avec constante et tendance</h4>
                <p>Utilisé pour les séries avec une tendance déterministe.</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Forme AR(1):**")
                    st.latex(r"""
                    X_t = c + \beta t + \phi X_{t-1} + \varepsilon_t
                    """)

                with col2:
                    st.markdown("**Forme DF:**")
                    st.latex(r"""
                    \Delta X_t = c + \beta t + \gamma X_{t-1} + \varepsilon_t
                    """)

                st.markdown("**Interprétation:**")
                st.markdown("""
                - **c**: constante (intercept)
                - **β**: coefficient de tendance déterministe
                - **γ**: coefficient de racine unitaire
                """)

                st.markdown("**Hypothèses:**")
                st.latex(r"""
                \begin{cases}
                H_0: \gamma = 0 & \text{(Racine unitaire avec tendance)} \\
                H_1: \gamma < 0 & \text{(Stationnaire autour d'une tendance)}
                \end{cases}
                """)

            st.markdown("---")
            st.markdown("### 📊 Tableau Récapitulatif")

            recap_df = pd.DataFrame({
                'Modèle': ['Modèle 1', 'Modèle 2', 'Modèle 3'],
                'Équation': [
                    'ΔXₜ = γXₜ₋₁ + εₜ',
                    'ΔXₜ = c + γXₜ₋₁ + εₜ',
                    'ΔXₜ = c + βt + γXₜ₋₁ + εₜ'
                ],
                'Usage': [
                    'Série autour de 0',
                    'Série avec moyenne',
                    'Série avec tendance'
                ],
                'H₀': ['γ = 0', 'γ = 0', 'γ = 0'],
                'H₁': ['γ < 0', 'γ < 0', 'γ < 0']
            })

            st.dataframe(recap_df, use_container_width=True)

        with tab3:
            st.markdown("### 💻 Application Pratique du Test DF")

            st.markdown("#### 📊 Génération de Données")

            col1, col2, col3 = st.columns(3)
            with col1:
                n_obs = st.slider("Nombre d'observations", 100, 500, 200, key='df_n')
            with col2:
                phi_val = st.slider("Valeur de φ", 0.5, 1.0, 0.95, 0.01, key='df_phi')
            with col3:
                const = st.slider("Constante (c)", 0.0, 5.0, 1.0, key='df_const')

            # Simulation
            np.random.seed(42)
            epsilon = np.random.normal(0, 1, n_obs)
            X = np.zeros(n_obs)
            X[0] = const

            for t in range(1, n_obs):
                X[t] = const + phi_val * X[t - 1] + epsilon[t]

            # Calcul des différences
            delta_X = np.diff(X)
            X_lag = X[:-1]

            # Régression
            from scipy import stats

            # Ajout de constante
            X_lag_with_const = np.column_stack([np.ones(len(X_lag)), X_lag])

            # Régression linéaire
            beta = np.linalg.lstsq(X_lag_with_const, delta_X, rcond=None)[0]
            c_hat = beta[0]
            gamma_hat = beta[1]

            # Résidus
            residuals = delta_X - (c_hat + gamma_hat * X_lag)

            # Erreur standard
            se_gamma = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2)) / np.sqrt(
                np.sum((X_lag - np.mean(X_lag)) ** 2))

            # Statistique DF
            df_stat = gamma_hat / se_gamma

            # Affichage des résultats
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Série Temporelle")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(y=X, mode='lines', name='Xₜ', line=dict(color='blue')))
                fig1.update_layout(title='Série Simulée', xaxis_title='Temps', yaxis_title='Valeur')
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("### 📉 Première Différence")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=delta_X, mode='lines', name='ΔXₜ', line=dict(color='red')))
                fig2.update_layout(title='Première Différence', xaxis_title='Temps', yaxis_title='ΔXₜ')
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 📊 Résultats du Test")

            results_df = pd.DataFrame({
                'Paramètre': ['Constante estimée (ĉ)', 'Coefficient estimé (γ̂)', 'Erreur standard SE(γ̂)',
                              'Statistique DF', 'φ estimé'],
                'Valeur': [c_hat, gamma_hat, se_gamma, df_stat, gamma_hat + 1]
            })

            st.dataframe(results_df.style.format({'Valeur': '{:.4f}'}), use_container_width=True)

            # Valeurs critiques (approximatives pour modèle 2)
            critical_values = {
                '1%': -3.43,
                '5%': -2.86,
                '10%': -2.57
            }

            st.markdown("### 🎯 Décision")

            st.markdown(f"""
            <div class="formula-box">
            <p><strong>Statistique DF calculée:</strong> {df_stat:.4f}</p>
            <p><strong>Valeurs critiques (Modèle 2):</strong></p>
            <ul>
                <li>1% : {critical_values['1%']}</li>
                <li>5% : {critical_values['5%']}</li>
                <li>10% : {critical_values['10%']}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            if df_stat < critical_values['5%']:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Conclusion: Rejet de H₀</h4>
                <p>La série est <strong>STATIONNAIRE</strong> au seuil de 5%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Conclusion: Non-rejet de H₀</h4>
                <p>La série contient une <strong>RACINE UNITAIRE</strong> (non-stationnaire)</p>
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            st.markdown("### 🎯 Guide d'Interprétation")

            st.markdown("""
            <div class="info-box">
            <h4>📋 Étapes d'Interprétation du Test DF</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 1️⃣ **Formulation des Hypothèses**")
            st.latex(r"""
            \begin{cases}
            H_0: \gamma = 0 & \text{(Présence de racine unitaire)} \\
            H_1: \gamma < 0 & \text{(Pas de racine unitaire)}
            \end{cases}
            """)

            st.markdown("#### 2️⃣ **Calcul de la Statistique de Test**")
            st.latex(r"""
            DF = \frac{\hat{\gamma}}{SE(\hat{\gamma})}
            """)

            st.markdown("#### 3️⃣ **Comparaison avec les Valeurs Critiques**")

            st.markdown("""
            <div class="warning-box">
            <p><strong>⚠️ Attention:</strong> Les valeurs critiques du test DF sont différentes
            de celles de la distribution de Student!</p>
            </div>
            """, unsafe_allow_html=True)

            # Tableau des valeurs critiques
            st.markdown("##### Valeurs Critiques de Dickey-Fuller")

            cv_df = pd.DataFrame({
                'Modèle': ['Modèle 1', 'Modèle 2', 'Modèle 3'],
                '1%': [-2.58, -3.43, -3.96],
                '5%': [-1.95, -2.86, -3.41],
                '10%': [-1.62, -2.57, -3.12]
            })

            st.dataframe(cv_df, use_container_width=True)

            st.markdown("#### 4️⃣ **Règle de Décision**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Si DF < Valeur Critique</h4>
                <p><strong>Rejet de H₀</strong></p>
                <p>➡️ Série STATIONNAIRE</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Si DF ≥ Valeur Critique</h4>
                <p><strong>Non-rejet de H₀</strong></p>
                <p>➡️ Série NON-STATIONNAIRE</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Exemples Pratiques")

            example = st.selectbox(
                "Choisissez un exemple:",
                ["Exemple 1: Série Stationnaire",
                 "Exemple 2: Marche Aléatoire",
                 "Exemple 3: Série avec Tendance"]
            )

            if example == "Exemple 1: Série Stationnaire":
                st.markdown("""
                **Données:** Prix d'un produit autour d'une moyenne stable

                **Résultat du test:**
                - Statistique DF = -4.25
                - Valeur critique (5%) = -2.86

                **Interprétation:**
                """)

                st.latex(r"-4.25 < -2.86 \Rightarrow \text{Rejet de } H_0")

                st.markdown("""
                <div class="success-box">
                <p><strong>Conclusion:</strong> La série est stationnaire.
                On peut utiliser directement des modèles ARMA.</p>
                </div>
                """, unsafe_allow_html=True)

            elif example == "Exemple 2: Marche Aléatoire":
                st.markdown("""
                **Données:** Prix d'une action en bourse

                **Résultat du test:**
                - Statistique DF = -1.52
                - Valeur critique (5%) = -2.86

                **Interprétation:**
                """)

                st.latex(r"-1.52 > -2.86 \Rightarrow \text{Non-rejet de } H_0")

                st.markdown("""
                <div class="warning-box">
                <p><strong>Conclusion:</strong> La série contient une racine unitaire.
                Il faut différencier la série avant modélisation.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                **Données:** PIB avec tendance croissante

                **Résultat du test (Modèle 3):**
                - Statistique DF = -2.95
                - Valeur critique (5%) = -3.41

                **Interprétation:**
                """)

                st.latex(r"-2.95 > -3.41 \Rightarrow \text{Non-rejet de } H_0")

                st.markdown("""
                <div class="info-box">
                <p><strong>Conclusion:</strong> Présence de racine unitaire avec tendance.
                Différenciation nécessaire ou retrait de la tendance.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### ⚠️ Limites du Test DF")

            st.markdown("""
            1. **Puissance faible** pour des échantillons de petite taille
            2. **Sensible à la spécification** du modèle (choix entre modèles 1, 2, 3)
            3. **Hypothèse** que les erreurs sont un bruit blanc (pas d'autocorrélation)
            4. **Solution:** Utiliser le test ADF (Augmented Dickey-Fuller) pour gérer l'autocorrélation
            """)

    # ========== SECTION 2.3 ==========
    elif section_ch2 == "2.3 - Test ADF (Augmented Dickey-Fuller)":
        st.markdown('<p class="sub-header">2.3 - Test ADF (Augmented Dickey-Fuller)</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📚 Théorie", "🔢 Modèles ADF", "💻 Application Python", "📊 Comparaison DF vs ADF"])

        with tab1:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Pourquoi le Test ADF?</h3>
            <p>Le test DF suppose que les erreurs sont un <strong>bruit blanc</strong> (pas d'autocorrélation).</p>
            <p>Le test ADF <strong>relaxe cette hypothèse</strong> en ajoutant des termes de retard.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📖 Problème du Test DF Simple")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="warning-box">
                <h4>❌ Test DF</h4>
                <p><strong>Hypothèse stricte:</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \varepsilon_t
                """)

                st.markdown("Où **εₜ ~ BB(0, σ²)** (bruit blanc)")

                st.markdown("""
                **Problème:** Si εₜ est autocorrélé, le test est biaisé!
                """)

            with col2:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Test ADF</h4>
                <p><strong>Hypothèse relaxée:</strong></p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \sum_{i=1}^{p} \beta_i \Delta X_{t-i} + \varepsilon_t
                """)

                st.markdown("Ajout de **p retards** de ΔXₜ pour capturer l'autocorrélation")

            st.markdown("---")
            st.markdown("### 🔬 Principe du Test ADF")

            st.markdown("""
            <div class="formula-box">
            <h4>L'idée clé:</h4>
            <p>Ajouter des termes de différences retardées pour \"nettoyer\" l'autocorrélation dans les résidus.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Modèle général ADF:**")
            st.latex(r"""
            \Delta X_t = c + \beta t + \gamma X_{t-1} + \beta_1 \Delta X_{t-1} + \beta_2 \Delta X_{t-2} + \cdots + \beta_p \Delta X_{t-p} + \varepsilon_t
            """)

            st.markdown("**Composantes:**")
            st.markdown("""
            - **c**: constante (drift)
            - **βt**: tendance déterministe (optionnelle)
            - **γXₜ₋₁**: terme de racine unitaire (à tester)
            - **βᵢΔXₜ₋ᵢ**: termes de correction pour autocorrélation
            - **εₜ**: terme d'erreur (bruit blanc)
            """)

            st.markdown("### 🎯 Hypothèses du Test")

            st.latex(r"""
            \begin{cases}
            H_0: \gamma = 0 & \text{(Racine unitaire - Série non-stationnaire)} \\
            H_1: \gamma < 0 & \text{(Pas de racine unitaire - Série stationnaire)}
            \end{cases}
            """)

            st.markdown("""
            <div class="warning-box">
            <p><strong>⚠️ Important:</strong> Les hypothèses sont les mêmes que pour le test DF,
            mais le test ADF est plus robuste!</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 🔢 Les Trois Modèles ADF")

            adf_model = st.radio(
                "Sélectionnez le modèle ADF:",
                ["Modèle 1: Sans constante ni tendance",
                 "Modèle 2: Avec constante",
                 "Modèle 3: Avec constante et tendance"],
                horizontal=False
            )

            if adf_model == "Modèle 1: Sans constante ni tendance":
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle ADF 1</h4>
                <p>Pour une série oscillant autour de zéro sans dérive ni tendance.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = \gamma X_{t-1} + \sum_{i=1}^{p} \beta_i \Delta X_{t-i} + \varepsilon_t
                """)

                st.markdown("**Forme développée (p=2):**")
                st.latex(r"""
                \Delta X_t = \gamma X_{t-1} + \beta_1 \Delta X_{t-1} + \beta_2 \Delta X_{t-2} + \varepsilon_t
                """)

            elif adf_model == "Modèle 2: Avec constante":
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle ADF 2</h4>
                <p>Pour une série avec une moyenne non nulle (présence de drift).</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \sum_{i=1}^{p} \beta_i \Delta X_{t-i} + \varepsilon_t
                """)

                st.markdown("**Forme développée (p=3):**")
                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \beta_1 \Delta X_{t-1} + \beta_2 \Delta X_{t-2} + \beta_3 \Delta X_{t-3} + \varepsilon_t
                """)

            else:  # Modèle 3
                st.markdown("""
                <div class="info-box">
                <h4>📐 Modèle ADF 3</h4>
                <p>Pour une série avec une tendance déterministe.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \beta_i \Delta X_{t-i} + \varepsilon_t
                """)

                st.markdown("**Forme développée (p=2):**")
                st.latex(r"""
                \Delta X_t = c + \beta t + \gamma X_{t-1} + \beta_1 \Delta X_{t-1} + \beta_2 \Delta X_{t-2} + \varepsilon_t
                """)

            st.markdown("---")
            st.markdown("### 🔍 Choix du Nombre de Retards (p)")

            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Question cruciale: Comment choisir p?</h4>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Méthodes courantes:**")

            methods_df = pd.DataFrame({
                'Méthode': ['AIC (Akaike)', 'BIC (Schwarz)', 'Règle empirique', 'ACF/PACF'],
                'Formule/Critère': [
                    'AIC = -2log(L) + 2k',
                    'BIC = -2log(L) + k·log(n)',
                    'p ≈ 12(n/100)^(1/4)',
                    'Analyse graphique'
                ],
                'Caractéristique': [
                    'Minimise AIC',
                    'Plus parcimonieux',
                    'Simple',
                    'Visuelle'
                ]
            })

            st.dataframe(methods_df, use_container_width=True)

            st.markdown("""
            **Recommandations:**
            - **AIC**: Tend à sélectionner plus de paramètres
            - **BIC**: Tend à sélectionner moins de paramètres (préféré pour grands échantillons)
            - **Pratique courante**: Tester plusieurs valeurs et comparer
            """)

        with tab3:
            st.markdown("### 💻 Application avec Python (statsmodels)")

            st.markdown("#### 📊 Génération de Données")

            col1, col2, col3 = st.columns(3)

            with col1:
                n_sample = st.slider("Nombre d'observations", 100, 500, 200, key='adf_n')
            with col2:
                phi_param = st.slider("φ (AR coefficient)", 0.7, 1.0, 0.98, 0.01, key='adf_phi')
            with col3:
                ar_order = st.slider("Ordre AR additionnel", 0, 3, 1, key='adf_ar')

            # Simulation d'un processus AR
            np.random.seed(42)

            # Générer un processus avec structure AR
            from scipy import signal

            # Coefficients AR
            ar_coef = [1, -phi_param]
            if ar_order > 0:
                for i in range(ar_order):
                    ar_coef.append(np.random.uniform(-0.2, 0.2))

            # Simulation
            white_noise = np.random.normal(0, 1, n_sample + 100)
            X_series = signal.lfilter([1], ar_coef, white_noise)[100:]

            # Affichage de la série
            st.markdown("#### 📈 Série Temporelle Générée")

            fig_series = go.Figure()
            fig_series.add_trace(go.Scatter(y=X_series, mode='lines', name='Série', line=dict(color='blue')))
            fig_series.update_layout(
                title=f'Série Simulée (φ={phi_param})',
                xaxis_title='Temps',
                yaxis_title='Valeur',
                height=400
            )
            st.plotly_chart(fig_series, use_container_width=True)

            # Application du test ADF
            st.markdown("#### 🧪 Test ADF avec statsmodels")

            # Choix des paramètres du test
            col1, col2 = st.columns(2)

            with col1:
                regression_type = st.selectbox(
                    "Type de régression:",
                    ['c', 'ct', 'ctt', 'n'],
                    format_func=lambda x: {
                        'c': 'Constante seule',
                        'ct': 'Constante + Tendance',
                        'ctt': 'Constante + Tendance + Tendance²',
                        'n': 'Aucune'
                    }[x]
                )

            with col2:
                max_lag = st.slider("Nombre maximum de retards", 0, 10, 5, key='adf_maxlag')

            # Code Python à afficher
            st.markdown("**Code Python:**")

            code = f"""
from statsmodels.tsa.stattools import adfuller

# Application du test ADF
result = adfuller(X_series,
                 maxlag={max_lag},
                 regression='{regression_type}',
                 autolag='AIC')

# Extraction des résultats
adf_statistic = result[0]
p_value = result[1]
used_lag = result[2]
n_obs = result[3]
critical_values = result[4]
ic_best = result[5]

print(f"Statistique ADF: {{adf_statistic:.4f}}")
print(f"p-value: {{p_value:.4f}}")
print(f"Nombre de retards utilisés: {{used_lag}}")
print(f"Nombre d'observations: {{n_obs}}")
print("Valeurs critiques:")
for key, value in critical_values.items():
    print(f"  {{key}}: {{value:.4f}}")
"""

            st.code(code, language='python')

            # Exécution réelle du test
            try:
                result = adfuller(X_series, maxlag=max_lag, regression=regression_type, autolag='AIC')

                adf_stat = result[0]
                p_val = result[1]
                used_lag = result[2]
                nobs = result[3]
                crit_vals = result[4]

                st.markdown("#### 📊 Résultats du Test ADF")

                # Métriques principales
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Statistique ADF", f"{adf_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{p_val:.4f}")
                with col3:
                    st.metric("Retards utilisés", used_lag)
                with col4:
                    st.metric("Observations", nobs)

                # Valeurs critiques
                st.markdown("**Valeurs Critiques:**")

                crit_df = pd.DataFrame({
                    'Niveau': ['1%', '5%', '10%'],
                    'Valeur Critique': [crit_vals['1%'], crit_vals['5%'], crit_vals['10%']],
                    'Test': [
                        '✅ Rejet' if adf_stat < crit_vals['1%'] else '❌ Non-rejet',
                        '✅ Rejet' if adf_stat < crit_vals['5%'] else '❌ Non-rejet',
                        '✅ Rejet' if adf_stat < crit_vals['10%'] else '❌ Non-rejet'
                    ]
                })

                st.dataframe(crit_df, use_container_width=True)

                # Interprétation
                st.markdown("#### 🎯 Interprétation")

                if p_val < 0.05:
                    st.markdown("""
                    <div class="success-box">
                    <h4>✅ Résultat: Série STATIONNAIRE</h4>
                    <p>La p-value est inférieure à 0.05, nous rejetons H₀.</p>
                    <p>La série ne contient pas de racine unitaire.</p>
                    <p><strong>→ La série peut être utilisée directement pour la modélisation ARMA.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                    <h4>⚠️ Résultat: Série NON-STATIONNAIRE</h4>
                    <p>La p-value est supérieure à 0.05, nous ne rejetons pas H₀.</p>
                    <p>La série contient une racine unitaire.</p>
                    <p><strong>→ Différenciation nécessaire avant modélisation.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Si non-stationnaire, montrer la série différenciée
                if p_val >= 0.05:
                    st.markdown("#### 📉 Série Après Différenciation")

                    X_diff = np.diff(X_series)

                    fig_diff = go.Figure()
                    fig_diff.add_trace(go.Scatter(y=X_diff, mode='lines', name='ΔX', line=dict(color='red')))
                    fig_diff.update_layout(
                        title='Série Différenciée',
                        xaxis_title='Temps',
                        yaxis_title='ΔX',
                        height=400
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)

                    # Test ADF sur série différenciée
                    result_diff = adfuller(X_diff, maxlag=max_lag, regression=regression_type, autolag='AIC')

                    st.markdown(f"""
                    **Test ADF sur la série différenciée:**
                    - Statistique ADF: {result_diff[0]:.4f}
                    - p-value: {result_diff[1]:.4f}
                    """)

                    if result_diff[1] < 0.05:
                        st.markdown("""
                        <div class="success-box">
                        <p>✅ La série différenciée est stationnaire!</p>
                        <p>Ordre d'intégration: <strong>I(1)</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erreur lors du test ADF: {str(e)}")

            # Guide d'utilisation
            st.markdown("---")
            st.markdown("### 📚 Guide d'Utilisation du Test ADF")

            with st.expander("🔍 Étapes Pratiques"):
                st.markdown("""
                **1. Visualiser la série**
                   - Graphique de la série temporelle
                   - ACF et PACF

                **2. Choisir le modèle approprié**
                   - 'c': série avec moyenne constante
                   - 'ct': série avec tendance
                   - 'n': série centrée autour de zéro

                **3. Appliquer le test ADF**
                   - Utiliser autolag='AIC' pour sélection automatique
                   - Ou spécifier maxlag manuellement

                **4. Interpréter les résultats**
                   - p-value < 0.05 → stationnaire
                   - p-value ≥ 0.05 → non-stationnaire

                **5. Si non-stationnaire**
                   - Différencier la série
                   - Retester avec ADF
                   - Répéter jusqu'à stationnarité
                """)

        with tab4:
            st.markdown("### 📊 Comparaison: Test DF vs Test ADF")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>📝 Test DF (Dickey-Fuller)</h4>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \varepsilon_t
                """)

                st.markdown("""
                **Avantages:**
                - ✅ Simple à comprendre
                - ✅ Peu de paramètres à estimer
                - ✅ Rapide à calculer

                **Inconvénients:**
                - ❌ Suppose εₜ ~ bruit blanc
                - ❌ Peu robuste à l'autocorrélation
                - ❌ Puissance faible si autocorrélation
                """)

            with col2:
                st.markdown("""
                <div class="success-box">
                <h4>📝 Test ADF (Augmented DF)</h4>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                \Delta X_t = c + \gamma X_{t-1} + \sum_{i=1}^{p}\beta_i \Delta X_{t-i} + \varepsilon_t
                """)

                st.markdown("""
                **Avantages:**
                - ✅ Robuste à l'autocorrélation
                - ✅ Plus de puissance statistique
                - ✅ Gère les structures AR complexes

                **Inconvénients:**
                - ❌ Choix de p (nombre de retards)
                - ❌ Plus complexe
                - ❌ Perte d'observations
                """)

            st.markdown("---")
            st.markdown("### 🔬 Simulation Comparative")

            # Paramètres de simulation
            col1, col2 = st.columns(2)

            with col1:
                n_sim = st.slider("Taille échantillon", 100, 500, 200, key='comp_n')
            with col2:
                autocorr_level = st.slider("Niveau d'autocorrélation", 0.0, 0.9, 0.5, key='comp_ar')

            # Générer processus avec autocorrélation
            np.random.seed(42)
            noise = np.random.normal(0, 1, n_sim)

            # AR(1) process
            x_ar = np.zeros(n_sim)
            for t in range(1, n_sim):
                x_ar[t] = autocorr_level * x_ar[t - 1] + noise[t]

            # Random walk (non-stationnaire)
            x_rw = np.cumsum(noise)

            # Tests
            from statsmodels.tsa.stattools import adfuller

            # DF simple (maxlag=0)
            df_result_ar = adfuller(x_ar, maxlag=0, regression='c')
            df_result_rw = adfuller(x_rw, maxlag=0, regression='c')

            # ADF (autolag)
            adf_result_ar = adfuller(x_ar, regression='c', autolag='AIC')
            adf_result_rw = adfuller(x_rw, regression='c', autolag='AIC')

            # Résultats
            st.markdown("#### 📊 Résultats de la Comparaison")

            results_comp = pd.DataFrame({
                'Série': ['AR(1) - Stationnaire', 'AR(1) - Stationnaire',
                          'Random Walk - Non-stat', 'Random Walk - Non-stat'],
                'Test': ['DF', 'ADF', 'DF', 'ADF'],
                'Statistique': [df_result_ar[0], adf_result_ar[0],
                                df_result_rw[0], adf_result_rw[0]],
                'p-value': [df_result_ar[1], adf_result_ar[1],
                            df_result_rw[1], adf_result_rw[1]],
                'Retards': [0, adf_result_ar[2], 0, adf_result_rw[2]],
                'Conclusion': [
                    '✅ Stationnaire' if df_result_ar[1] < 0.05 else '❌ Non-stat',
                    '✅ Stationnaire' if adf_result_ar[1] < 0.05 else '❌ Non-stat',
                    '✅ Stationnaire' if df_result_rw[1] < 0.05 else '❌ Non-stat',
                    '✅ Stationnaire' if adf_result_rw[1] < 0.05 else '❌ Non-stat'
                ]
            })

            st.dataframe(results_comp.style.format({
                'Statistique': '{:.4f}',
                'p-value': '{:.4f}'
            }), use_container_width=True)

            st.markdown("""
            <div class="info-box">
            <h4>💡 Observations:</h4>
            <ul>
                <li>Le test ADF est généralement plus fiable en présence d'autocorrélation</li>
                <li>Pour les séries simples, DF et ADF donnent des résultats similaires</li>
                <li>Le test ADF ajuste automatiquement le nombre de retards nécessaires</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            # Recommandations
            st.markdown("---")
            st.markdown("### 💡 Recommandations Pratiques")

            st.markdown("""
            | Situation | Test Recommandé | Raison |
            |-----------|----------------|---------|
            | Série simple, pas d'autocorrélation évidente | **DF** | Plus simple, suffisant |
            | Autocorrélation présente (ACF significatif) | **ADF** | Corrige l'autocorrélation |
            | Doute sur la structure | **ADF** | Plus robuste, sélection auto des retards |
            | Analyse professionnelle | **ADF** | Standard de l'industrie |
            | Données financières | **ADF** | Structure souvent complexe |
            """)

            st.markdown("""
            <div class="success-box">
            <h4>✅ Meilleure Pratique:</h4>
            <p><strong>Utiliser toujours le test ADF</strong> sauf si vous avez une raison spécifique
            d'utiliser le test DF simple.</p>
            </div>
            """, unsafe_allow_html=True)

    # ========== SECTION 2.4 ==========
    elif section_ch2 == "2.4 - Processus ARIMA":
        st.markdown('<p class="sub-header">2.4 - Processus ARIMA</p>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 Théorie",
            "🔢 Composantes ARIMA",
            "🎯 Identification",
            "💻 Application",
            "📊 Exemples Réels"
        ])

        with tab1:
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Qu'est-ce qu'un Processus ARIMA?</h3>
            <p><strong>ARIMA</strong> = AutoRegressive Integrated Moving Average</p>
            <p>Un modèle complet pour modéliser des séries temporelles <strong>non-stationnaires</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📖 Définition")

            st.markdown("""
            <div class="formula-box">
            <p>Un processus ARIMA(p, d, q) combine:</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="success-box">
                <h4>AR(p)</h4>
                <p><strong>AutoRegressive</strong></p>
                <p>Dépendance aux valeurs passées</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="info-box">
                <h4>I(d)</h4>
                <p><strong>Integrated</strong></p>
                <p>Ordre de différenciation</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="warning-box">
                <h4>MA(q)</h4>
                <p><strong>Moving Average</strong></p>
                <p>Dépendance aux erreurs passées</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📐 Notation ARIMA(p, d, q)")

            st.markdown("""
            - **p** : Ordre de la partie AutoRégressive (AR)
            - **d** : Ordre d'intégration (nombre de différenciations)
            - **q** : Ordre de la partie Moyenne Mobile (MA)
            """)

            st.markdown("### 🔄 Du Non-Stationnaire au Stationnaire")

            st.markdown("""
            <div class="formula-box">
            <h4>Processus Général:</h4>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r"""
            X_t \xrightarrow{\text{Différenciation } d \text{ fois}} Y_t \sim ARMA(p, q)
            """)

            st.markdown("**Étapes:**")

            st.latex(r"""
            \begin{align}
            &\text{1. Série originale: } X_t \text{ (non-stationnaire)} \\
            &\text{2. Différenciation: } Y_t = \Delta^d X_t = (1-L)^d X_t \\
            &\text{3. Modèle ARMA: } \phi(L) Y_t = \theta(L) \varepsilon_t
            \end{align}
            """)

            st.markdown("---")
            st.markdown("### 📊 Cas Particuliers")

            cases_df = pd.DataFrame({
                'Modèle': ['ARIMA(p,0,0)', 'ARIMA(0,0,q)', 'ARIMA(p,0,q)', 'ARIMA(0,1,0)', 'ARIMA(0,d,0)'],
                'Équivalent': ['AR(p)', 'MA(q)', 'ARMA(p,q)', 'Random Walk', 'I(d)'],
                'Description': [
                    'Processus purement autorégressif',
                    'Processus purement moyenne mobile',
                    'ARMA stationnaire',
                    'Marche aléatoire',
                    'Processus intégré d ordre'
                ]
            })

            st.dataframe(cases_df, use_container_width=True)

        with tab2:
            st.markdown("### 🔢 Détails des Composantes ARIMA")

            component = st.selectbox(
                "Choisissez une composante à explorer:",
                ["Composante AR (p)", "Composante I (d)", "Composante MA (q)", "Combinaison Complète"]
            )

            if component == "Composante AR (p)":
                st.markdown("""
                <div class="info-box">
                <h4>📈 Partie AutoRégressive AR(p)</h4>
                <p>La valeur actuelle dépend de ses <strong>p valeurs passées</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t
                """)

                st.markdown("**Forme opérateur:**")
                st.latex(r"""
                \phi(L) X_t = \varepsilon_t
                """)

                st.markdown("Où:")
                st.latex(r"""
                \phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p
                """)

                st.markdown("**Exemples:**")

                with st.expander("AR(1)"):
                    st.latex(r"X_t = \phi_1 X_{t-1} + \varepsilon_t")
                    st.markdown("- Processus de premier ordre")
                    st.markdown("- Mémoire courte")
                    st.markdown("- Stationnaire si |φ₁| < 1")

                with st.expander("AR(2)"):
                    st.latex(r"X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \varepsilon_t")
                    st.markdown("- Peut générer des cycles")
                    st.markdown("- Conditions de stationnarité plus complexes")

                # PACF
                st.markdown("**📊 Identification via PACF:**")
                st.markdown("""
                - PACF se coupe après le retard **p**
                - ACF décroît exponentiellement
                """)

            elif component == "Composante I (d)":
                st.markdown("""
                <div class="info-box">
                <h4>🔄 Partie Intégration I(d)</h4>
                <p>Nombre de <strong>différenciations</strong> nécessaires pour rendre la série stationnaire.</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Opérateur de différence:**")
                st.latex(r"""
                \Delta X_t = X_t - X_{t-1} = (1-L) X_t
                """)

                st.markdown("**Différenciation d'ordre d:**")
                st.latex(r"""
                \Delta^d X_t = (1-L)^d X_t
                """)

                st.markdown("**Exemples:**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**d = 0**")
                    st.latex(r"Y_t = X_t")
                    st.markdown("Série déjà stationnaire")

                with col2:
                    st.markdown("**d = 1**")
                    st.latex(r"Y_t = \Delta X_t")
                    st.markdown("Une différenciation")

                with col3:
                    st.markdown("**d = 2**")
                    st.latex(r"Y_t = \Delta^2 X_t")
                    st.markdown("Deux différenciations")

                st.markdown("---")
                st.markdown("**📊 Détermination de d:**")

                st.markdown("""
                1. **Test ADF** sur la série originale
                   - Si non-stationnaire → différencier
                2. **Test ADF** sur la série différenciée
                   - Si stationnaire → d = 1
                   - Si non → continuer
                3. **Répéter** jusqu'à stationnarité

                <div class="warning-box">
                <p><strong>⚠️ Attention:</strong> Rarement d > 2 dans la pratique!</p>
                </div>
                """, unsafe_allow_html=True)

            elif component == "Composante MA (q)":
                st.markdown("""
                <div class="info-box">
                <h4>📉 Partie Moyenne Mobile MA(q)</h4>
                <p>La valeur actuelle dépend des <strong>q erreurs passées</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                st.latex(r"""
                X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
                """)

                st.markdown("**Forme opérateur:**")
                st.latex(r"""
                X_t = \theta(L) \varepsilon_t
                """)

                st.markdown("Où:")
                st.latex(r"""
                \theta(L) = 1 + \theta_1 L + \theta_2 L^2 + \cdots + \theta_q L^q
                """)

                st.markdown("**Exemples:**")

                with st.expander("MA(1)"):
                    st.latex(r"X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1}")
                    st.markdown("- Mémoire très courte (1 période)")
                    st.markdown("- Toujours stationnaire")
                    st.markdown("- Invertible si |θ₁| < 1")

                with st.expander("MA(2)"):
                    st.latex(r"X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2}")
                    st.markdown("- Mémoire de 2 périodes")
                    st.markdown("- Conditions d'invertibilité plus complexes")

                # ACF
                st.markdown("**📊 Identification via ACF:**")
                st.markdown("""
                - ACF se coupe après le retard **q**
                - PACF décroît exponentiellement
                """)

            else:  # Combinaison complète
                st.markdown("""
                <div class="success-box">
                <h4>🎯 Modèle ARIMA(p,d,q) Complet</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📐 Formulation Mathématique")

                st.latex(r"""
                \phi(L)(1-L)^d X_t = \theta(L) \varepsilon_t
                """)

                st.markdown("**Développé:**")
                st.latex(r"""
                (1 - \phi_1 L - \cdots - \phi_p L^p)(1-L)^d X_t = (1 + \theta_1 L + \cdots + \theta_q L^q) \varepsilon_t
                """)

                st.markdown("---")
                st.markdown("### 🔍 Exemple: ARIMA(1,1,1)")

                st.markdown("**Étape 1: Différenciation**")
                st.latex(r"Y_t = \Delta X_t = X_t - X_{t-1}")

                st.markdown("**Étape 2: Modèle ARMA(1,1) sur Yₜ**")
                st.latex(r"Y_t = \phi_1 Y_{t-1} + \varepsilon_t + \theta_1 \varepsilon_{t-1}")

                st.markdown("**Forme développée:**")
                st.latex(r"""
                X_t - X_{t-1} = \phi_1(X_{t-1} - X_{t-2}) + \varepsilon_t + \theta_1 \varepsilon_{t-1}
                """)

                st.latex(r"""
                X_t = (1+\phi_1)X_{t-1} - \phi_1 X_{t-2} + \varepsilon_t + \theta_1 \varepsilon_{t-1}
                """)

        with tab3:
            st.markdown("### 🎯 Méthodologie d'Identification ARIMA")

            st.markdown("""
            <div class="info-box">
            <h4>📋 Approche de Box-Jenkins</h4>
            <p>Méthodologie systématique en 4 étapes pour identifier un modèle ARIMA.</p>
            </div>
            """, unsafe_allow_html=True)

            # Étapes
            step = st.radio(
                "Sélectionnez une étape:",
                ["Étape 1: Identification de d",
                 "Étape 2: Identification de p et q",
                 "Étape 3: Estimation",
                 "Étape 4: Validation"],
                horizontal=False
            )

            if step == "Étape 1: Identification de d":
                st.markdown("""
                <div class="formula-box">
                <h4>🔍 Déterminer l'Ordre de Différenciation (d)</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Méthode 1: Test ADF**")

                st.markdown(r"""
                ```
                1. Appliquer le test ADF sur Xₜ
                   - Si stationnaire → d = 0
                   - Sinon → continuer

                2. Calculer ΔXₜ et appliquer ADF
                   - Si stationnaire → d = 1
                   - Sinon → continuer

                3. Calculer Δ²Xₜ et appliquer ADF
                   - Si stationnaire → d = 2
                ```
                """)

                st.markdown("**Méthode 2: Analyse ACF**")

                st.markdown("""
                - **d = 0**: ACF décroît rapidement vers 0
                - **d = 1**: ACF décroît très lentement (proche de 1)
                - **d = 2**: ACF décroît linéairement
                """)

                st.markdown("**Méthode 3: Visualisation**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                    **Série nécessitant d=0:**
                    - Oscille autour d'une moyenne
                    - Variance constante
                    - Pas de tendance claire
                    """)

                with col2:
                    st.markdown("""
                    **Série nécessitant d≥1:**
                    - Tendance claire
                    - Variance croissante
                    - Pas de réversion à la moyenne
                    """)

            elif step == "Étape 2: Identification de p et q":
                st.markdown("""
                <div class="formula-box">
                <h4>🔍 Déterminer les Ordres p et q</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Sur la série stationnaire Yₜ = Δᵈ Xₜ:**")

                st.markdown("### 📊 Analyse ACF et PACF")

                # Tableau de décision
                decision_df = pd.DataFrame({
                    'Modèle': ['AR(p)', 'MA(q)', 'ARMA(p,q)'],
                    'ACF': [
                        'Décroît exponentiellement ou sinusoïdalement',
                        'Se coupe après le retard q',
                        'Décroît exponentiellement'
                    ],
                    'PACF': [
                        'Se coupe après le retard p',
                        'Décroît exponentiellement ou sinusoïdalement',
                        'Décroît exponentiellement'
                    ],
                    'Identification': [
                        'p = dernier pic significatif du PACF',
                        'q = dernier pic significatif de l\'ACF',
                        'Plusieurs modèles possibles'
                    ]
                })

                st.dataframe(decision_df, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📈 Critères d'Information")

                st.markdown("""
                Quand plusieurs modèles sont possibles, comparer avec:
                """)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**AIC (Akaike)**")
                    st.latex(r"AIC = -2\ln(L) + 2k")
                    st.markdown("- k = nombre de paramètres")
                    st.markdown("- Choisir le modèle avec AIC minimal")

                with col2:
                    st.markdown("**BIC (Bayésien)**")
                    st.latex(r"BIC = -2\ln(L) + k\ln(n)")
                    st.markdown("- n = nombre d'observations")
                    st.markdown("- Pénalise plus les modèles complexes")

            elif step == "Étape 3: Estimation":
                st.markdown("""
                <div class="formula-box">
                <h4>🔍 Estimation des Paramètres</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Méthodes d'Estimation:**")

                st.markdown("### 1️⃣ Maximum de Vraisemblance (ML)")

                st.latex(r"""
                \hat{\theta} = \arg\max L(\theta | X_1, \ldots, X_n)
                """)

                st.markdown("- Méthode la plus utilisée")
                st.markdown("- Propriétés asymptotiques optimales")
                st.markdown("- Implémentation dans statsmodels, R, etc.")

                st.markdown("### 2️⃣ Moindres Carrés (LS)")

                st.latex(r"""
                \hat{\theta} = \arg\min \sum_{t=1}^{n} \varepsilon_t^2
                """)

                st.markdown("- Plus simple computationnellement")
                st.markdown("- Équivalent à ML pour modèles gaussiens")

                st.markdown("---")
                st.markdown("### 📊 Vérification des Paramètres")

                st.markdown("""
                **Tests de significativité:**
                """)

                st.latex(r"""
                t = \frac{\hat{\phi}_i}{SE(\hat{\phi}_i)} \sim t_{n-k}
                """)

                st.markdown("""
                - Si |t| > valeur critique → paramètre significatif
                - Sinon → envisager un modèle plus parcimonieux
                """)

            else:  # Étape 4
                st.markdown("""
                <div class="formula-box">
                <h4>🔍 Validation du Modèle (Diagnostic)</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### ✅ Tests sur les Résidus")

                st.markdown("**1. Test de Blancheur (Ljung-Box)**")

                st.latex(r"""
                Q = n(n+2)\sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2_{h-p-q}
                """)

                st.markdown("""
                - H₀: Les résidus sont un bruit blanc
                - Si p-value > 0.05 → résidus non corrélés ✅
                """)

                st.markdown("**2. Test de Normalité (Jarque-Bera)**")

                st.latex(r"""
                JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2_2
                """)

                st.markdown("- S: skewness (asymétrie)")
                st.markdown("- K: kurtosis (aplatissement)")

                st.markdown("**3. Tests d'Homoscédasticité**")

                st.markdown("""
                - Test ARCH
                - Test de Breusch-Pagan
                - Graphique des résidus au carré
                """)

                st.markdown("---")
                st.markdown("### 📊 Critères de Validation")

                validation_criteria = pd.DataFrame({
                    'Critère': [
                        'Ljung-Box p-value',
                        'Normalité des résidus',
                        'AIC/BIC',
                        'R² ajusté',
                        'Graphiques résiduels'
                    ],
                    'Bon Modèle': [
                        '> 0.05',
                        'p-value > 0.05',
                        'Minimal',
                        'Élevé',
                        'Pas de pattern'
                    ],
                    'Action si Mauvais': [
                        'Augmenter p ou q',
                        'Transformation données',
                        'Essayer autre modèle',
                        'Réviser modèle',
                        'Réviser spécification'
                    ]
                })

                st.dataframe(validation_criteria, use_container_width=True)

            # Flowchart
            st.markdown("---")
            st.markdown("### 🔄 Diagramme de la Méthodologie")

            st.markdown(r"""
            ```
            1. VISUALISATION
                ↓
            2. TEST DE STATIONNARITÉ (ADF)
                ↓
            3a. Si Stationnaire        3b. Si Non-Stationnaire
                → d = 0                     → Différencier
                ↓                           ↓
            4. ANALYSE ACF/PACF             Retour à (2)
                ↓
            5. PROPOSITION DE MODÈLES (p,q)
                ↓
            6. ESTIMATION DES PARAMÈTRES
                ↓
            7. DIAGNOSTIC DES RÉSIDUS
                ↓
            8a. Résidus OK             8b. Résidus Problématiques
                → VALIDATION                → Retour à (5)
                ↓
            9. PRÉVISION
            ```
            """)

        with tab4:
            st.markdown("### 💻 Application Complète ARIMA avec Python")

            # Options de données
            data_source = st.radio(
                "Source de données:",
                ["Données Simulées", "Télécharger vos données"],
                horizontal=True
            )

            series_data = None  # Initialize to None

            if data_source == "Données Simulées":
                st.markdown("#### 🎲 Simulation ARIMA")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    sim_n = st.slider("Observations", 100, 500, 200, key='arima_sim_n')
                with col2:
                    sim_p = st.slider("p", 0, 3, 1, key='arima_p')
                with col3:
                    sim_d = st.slider("d", 0, 2, 1, key='arima_d')
                with col4:
                    sim_q = st.slider("q", 0, 3, 1, key='arima_q')

                # Simulation
                np.random.seed(42)

                # Paramètres AR et MA
                ar_params = np.array([1] + [-0.5] * sim_p if sim_p > 0 else [1])
                ma_params = np.array([1] + [0.3] * sim_q if sim_q > 0 else [1])

                # Générer ARMA
                arma_process = ArmaProcess(ar_params, ma_params)
                y_stationary = arma_process.generate_sample(nsample=sim_n + 100)[100:]

                # Intégration (si d > 0)
                y_series = y_stationary.copy()
                for _ in range(sim_d):
                    y_series = np.cumsum(y_series) + np.random.normal(0, 0.1)  # Add small drift

                series_data = pd.Series(y_series)

            else:
                uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=['csv'])

                if uploaded_file is not None:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.dataframe(df_uploaded.head())

                    col_name = st.selectbox("Sélectionnez la colonne:", df_uploaded.columns)
                    if col_name:
                        series_data = pd.to_numeric(df_uploaded[col_name], errors='coerce').dropna()
                else:
                    st.warning("Veuillez télécharger un fichier CSV")

            if series_data is not None and not series_data.empty:
                # Visualisation
                st.markdown("#### 📈 Série Temporelle")

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(y=series_data, mode='lines', name='Série'))
                fig_ts.update_layout(title='Série Temporelle', xaxis_title='Temps', yaxis_title='Valeur', height=400)
                st.plotly_chart(fig_ts, use_container_width=True)

                # Analyse de stationnarité
                st.markdown("#### 🔍 Analyse de Stationnarité")

                adf_result = adfuller(series_data)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Statistique ADF", f"{adf_result[0]:.4f}")
                    st.metric("p-value", f"{adf_result[1]:.4f}")

                with col2:
                    if adf_result[1] < 0.05:
                        st.success("✅ Série STATIONNAIRE")
                        suggested_d = 0
                    else:
                        st.warning("⚠️ Série NON-STATIONNAIRE")
                        suggested_d = 1

                # ACF et PACF
                st.markdown("#### 📊 ACF et PACF")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                plt.suptitle(f'ACF/PACF de la série (différenciée {suggested_d} fois)', size=16)

                # Série différenciée si nécessaire
                if suggested_d > 0:
                    series_diff = series_data.diff().dropna()
                else:
                    series_diff = series_data

                plot_acf(series_diff, lags=min(40, len(series_diff) // 2 - 1), ax=ax1)
                plot_pacf(series_diff, lags=min(40, len(series_diff) // 2 - 1), ax=ax2)

                st.pyplot(fig)

                # Suggestion de modèle
                st.markdown("#### 🎯 Suggestion de Modèle")

                st.info(f"""
                **Analyse automatique:**
                - Ordre de différenciation suggéré: **d = {suggested_d}**
                - Examinez l'ACF et PACF ci-dessus pour identifier p et q.
                """)

                # Estimation du modèle
                st.markdown("#### 📊 Estimation du Modèle ARIMA")

                col1, col2, col3 = st.columns(3)

                with col1:
                    order_p = st.number_input("p", 0, 5, 1, key='final_p')
                with col2:
                    order_d = st.number_input("d", 0, 2, suggested_d, key='final_d')
                with col3:
                    order_q = st.number_input("q", 0, 5, 1, key='final_q')

                if st.button("🚀 Estimer le Modèle"):
                    with st.spinner("Estimation en cours..."):
                        try:
                            # Ajuster le modèle
                            model = ARIMA(series_data, order=(order_p, order_d, order_q))
                            fitted_model = model.fit()

                            # Résumé
                            st.markdown("##### 📋 Résumé du Modèle")
                            st.text(fitted_model.summary())

                            # Diagnostic des résidus
                            st.markdown("##### 🔍 Diagnostic des Résidus")

                            fig = fitted_model.plot_diagnostics(figsize=(12, 8))
                            st.pyplot(fig)

                            # Test de Ljung-Box
                            lb_test = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)

                            st.markdown("##### 📊 Test de Ljung-Box")
                            st.dataframe(lb_test)

                            if (lb_test['lb_pvalue'] > 0.05).all():
                                st.success("✅ Les résidus sont un bruit blanc (p-values > 0.05)")
                            else:
                                st.warning("⚠️ Autocorrélation résiduelle détectée")

                            # Prévision
                            st.markdown("##### 🔮 Prévisions")

                            n_forecast = st.slider("Nombre de périodes à prévoir", 1, 50, 10)

                            forecast_obj = fitted_model.get_forecast(steps=n_forecast)
                            forecast = forecast_obj.predicted_mean
                            conf_int = forecast_obj.conf_int()

                            fig_forecast = go.Figure()

                            # Données historiques
                            fig_forecast.add_trace(go.Scatter(
                                y=series_data,
                                mode='lines',
                                name='Données',
                                line=dict(color='blue')
                            ))

                            # Prévisions
                            forecast_index = pd.RangeIndex(start=len(series_data), stop=len(series_data) + n_forecast)
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast,
                                mode='lines',
                                name='Prévisions',
                                line=dict(color='red', dash='dash')
                            ))

                            # Intervalle de confiance
                            fig_forecast.add_trace(go.Scatter(
                                x=pd.concat([pd.Series(forecast_index), pd.Series(forecast_index[::-1])]),
                                y=pd.concat([conf_int.iloc[:, 1], conf_int.iloc[:, 0][::-1]]),
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='IC 95%'
                            ))

                            fig_forecast.update_layout(
                                title='Série et Prévisions',
                                xaxis_title='Temps',
                                yaxis_title='Valeur',
                                height=500
                            )
                            st.plotly_chart(fig_forecast, use_container_width=True)

                            # Valeurs de prévision
                            st.markdown("**Valeurs Prévues:**")
                            forecast_df = pd.DataFrame({
                                'Période': forecast_index,
                                'Prévision': forecast
                            })
                            st.dataframe(forecast_df)

                        except Exception as e:
                            st.error(f"Erreur lors de l'estimation: {str(e)}")

        with tab5:
            st.markdown("### 📊 Exemples d'Application Réels")

            example_type = st.selectbox(
                "Choisissez un domaine d'application:",
                ["Finance - Prix d'Actions",
                 "Économie - Taux de Chômage",
                 "Météo - Températures",
                 "Ventes - Données Commerciales"]
            )

            if example_type == "Finance - Prix d'Actions":
                st.markdown("""
                <div class="info-box">
                <h4>💹 Modélisation des Prix d'Actions</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Caractéristiques typiques:**
                - Série **non-stationnaire** (tendance stochastique)
                - Variance souvent non-constante (hétéroscédasticité)
                - Modèle typique: **ARIMA(1,1,1)** ou **ARIMA(2,1,2)**
                """)

                st.markdown("**Modèle commun:**")
                st.latex(r"""
                \text{Log}(P_t) - \text{Log}(P_{t-1}) = r_t \sim ARMA(p,q)
                """)

                st.markdown("**Exemple:**")
                st.code("""
# Simulation prix d'action
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 252)  # Rendements journaliers
prices = 100 * np.exp(np.cumsum(returns))  # Prix

# Modèle sur rendements logarithmiques
log_returns = np.diff(np.log(prices))

# ARIMA(1,1,1) sur prix ou ARMA(1,1) sur rendements
model = ARIMA(prices, order=(1,1,1))
""", language='python')

            elif example_type == "Économie - Taux de Chômage":
                st.markdown("""
                <div class="info-box">
                <h4>📉 Modélisation du Taux de Chômage</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Caractéristiques typiques:**
                - Persistance forte (autocorrélation)
                - Saisonnalité possible
                - Modèle typique: **ARIMA(2,1,0)** ou **SARIMA**
                """)

                st.markdown("**Approche:**")
                st.markdown("""
                1. Test ADF → généralement I(1)
                2. Différenciation: Δ(Taux)
                3. AR(2) souvent suffisant pour la partie stationnaire
                """)

            elif example_type == "Météo - Températures":
                st.markdown("""
                <div class="info-box">
                <h4>🌡️ Modélisation des Températures</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Caractéristiques typiques:**
                - **Forte saisonnalité** (annuelle)
                - Tendance long-terme possible (réchauffement)
                - Modèle typique: **SARIMA(1,0,1)(1,1,1)₁₂**
                """)

                st.markdown("**Modèle saisonnier:**")
                st.latex(r"""
                \phi(L)\Phi(L^{12})(1-L^{12})X_t = \theta(L)\Theta(L^{12})\varepsilon_t
                """)

            else:  # Ventes
                st.markdown("""
                <div class="info-box">
                <h4>🛒 Modélisation des Ventes</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                **Caractéristiques typiques:**
                - Tendance croissante
                - Saisonnalité (mensuelle, trimestrielle)
                - Événements spéciaux (promotions, fêtes)
                - Modèle typique: **SARIMA(1,1,1)(0,1,1)₁₂**
                """)

                st.markdown("**Prétraitement:**")
                st.markdown("""
                1. Transformation log pour stabiliser la variance
                2. Différenciation saisonnière
                3. Différenciation simple si nécessaire
                4. Modèle ARMA sur la série transformée
                """)

                st.code("""
# Exemple ventes mensuelles
# sales = pd.read_csv('sales.csv', parse_dates=['date'], index_col='date')

# Transformation log
# log_sales = np.log(sales)

# Modèle SARIMA
# model = SARIMAX(log_sales,
#                 order=(1,1,1),           # (p,d,q)
#                 seasonal_order=(0,1,1,12)) # (P,D,Q,s)

# fitted = model.fit()
# forecast = fitted.forecast(steps=12)

# Retransformation
# forecast_original = np.exp(forecast)
""", language='python')

            st.markdown("---")
            st.markdown("### 📋 Résumé des Bonnes Pratiques")

            st.markdown("""
            <div class="success-box">
            <h4>✅ Checklist ARIMA</h4>
            <ol>
                <li>📊 <strong>Visualiser</strong> la série (tendance, saisonnalité)</li>
                <li>🔍 <strong>Tester</strong> la stationnarité (ADF)</li>
                <li>🔄 <strong>Différencier</strong> si nécessaire</li>
                <li>📈 <strong>Analyser</strong> ACF/PACF</li>
                <li>🎯 <strong>Identifier</strong> p et q</li>
                <li>💻 <strong>Estimer</strong> plusieurs modèles</li>
                <li>📊 <strong>Comparer</strong> AIC/BIC</li>
                <li>✅ <strong>Valider</strong> résidus</li>
                <li>🔮 <strong>Prévoir</strong> et évaluer</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)