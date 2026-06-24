import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, make_scorer
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import time

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    confusion_matrix, roc_curve, roc_auc_score
)
def evaluate_model(model, X_test, y_test, model_name, metric="recall"):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # ROC + AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)   # <— nom changé pour éviter conflit

    # Choix du critère
    if metric == "recall":
        best_metric = recall
        metric_name = "Rappel"
    elif metric == "precision":
        best_metric = precision
        metric_name = "Précision"
    elif metric == "auc":
        best_metric = auc_value
        metric_name = "AUC"
    else:
        raise ValueError("metric doit être 'recall', 'precision' ou 'auc'")

    final_score = (accuracy + best_metric) / 2

    # --- Affichage graphique ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de confusion - {model_name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")

    plt.subplot(1,2,2)
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"Courbe ROC - {model_name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Résultats numériques ---
    print(f"\n=== {model_name} ===")
    print(f"Accuracy   : {accuracy:.3f}")
    print(f"Rappel     : {recall:.3f}")
    print(f"Précision  : {precision:.3f}")
    print(f"AUC        : {auc_value:.3f}")
    print(f"Critère choisi : {metric_name}")
    print(f"Score final = ({accuracy:.3f} + {best_metric:.3f}) / 2 = {final_score:.3f}\n")

    return final_score

# ============================================================
# 1️ FONCTION : Exécuter les classifieurs CART / kNN / MLP
# ============================================================

def run_classifiers_train_test(models, X_train, y_train, X_test, y_test, metric="recall"):

    scores = {}

    for name, model in models.items():
        print(f"\n🔷 Entraînement du modèle : {name}")
        model.fit(X_train, y_train)

        score = evaluate_model(model, X_test, y_test, name, metric)
        scores[name] = score

    print("\n===== Résultats finaux =====")
    for name, score in scores.items():
        print(f"{name:10s} → Score final : {score:.3f}")

    best = max(scores, key=scores.get)
    print(f"\n Meilleur modèle : {best} (score = {scores[best]:.3f})")

    return models[best]


#Normalisation 

from sklearn.preprocessing import StandardScaler

def normalize_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm, scaler


# ============================================================
#  Fonction : Application de l'ACP et création de nouvelles variables
# ============================================================

from sklearn.decomposition import PCA
import numpy as np

def apply_pca_and_extend(X_train_norm, X_test_norm, n_components=3, random_state=1):


    # 1) Création du modèle PCA
    pca = PCA(n_components=n_components, random_state=random_state)

    # 2) Fit uniquement sur l'entraînement
    X_train_pca = pca.fit_transform(X_train_norm)

    # 3) Transformation du test
    X_test_pca = pca.transform(X_test_norm)

    # 4) Concaténation des données normalisées + ACP
    X_train_extended = np.hstack((X_train_norm, X_train_pca))
    X_test_extended  = np.hstack((X_test_norm, X_test_pca))

    return X_train_extended, X_test_extended, pca



# ============================================================
#  Importance des variables avec RandomForest
# ============================================================

def variable_importance(Xtrain, Ytrain, nom_cols):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import matplotlib.pyplot as plt

    clf = RandomForestClassifier(n_estimators=1000, random_state=1)
    clf.fit(Xtrain, Ytrain)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    sorted_idx = np.argsort(importances)[::-1]

    print("=== Variables triées par importance ===\n")
    for i, idx in enumerate(sorted_idx):
        print(f"{i+1}. {nom_cols[idx]} : {round(importances[idx],4)}")

    padding = np.arange(len(importances)) + 0.5

    plt.figure(figsize=(10,8))
    plt.barh(padding, importances[sorted_idx], xerr=std[sorted_idx], align='center')
    plt.yticks(padding, nom_cols[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance (Random Forest)")
    plt.gca().invert_yaxis()
    plt.show()

    return sorted_idx


#select nb variables

def select_nb_variables(Xtrain, Xtest, Ytrain, Ytest, sorted_idx):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt

    KNN = KNeighborsClassifier(n_neighbors=5)

    scores = np.zeros(Xtrain.shape[1])

    for f in range(Xtrain.shape[1]):
        X1_f = Xtrain[:, sorted_idx[:f+1]]
        X2_f = Xtest[:, sorted_idx[:f+1]]

        KNN.fit(X1_f, Ytrain)
        YKNN = KNN.predict(X2_f)

        scores[f] = np.round(accuracy_score(Ytest, YKNN), 3)

    plt.figure(figsize=(9,4))
    plt.plot(scores, marker='o')
    plt.xlabel("Nombre de Variables Conservées")
    plt.ylabel("Accuracy")
    plt.title("Évolution de l'Accuracy en fonction du nombre de variables")
    plt.grid()
    plt.show()

    best_f = np.argmax(scores) + 1
    print(f" Nombre optimal de variables = {best_f} (Accuracy = {scores[best_f-1]})")

    return best_f, scores


# ============================================================
#  Optimisation des hyperparamètres MLP (GridSearch)
# ============================================================
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.neural_network import MLPClassifier
import numpy as np

# ==========================================================
#  Fonction : Optimisation des hyperparamètres du MLP
# ==========================================================

def tune_mlp_hyperparameters(Xtrain, Ytrain):
    """
    Tune les hyperparamètres du MLP à l'aide de GridSearchCV.
    Le critère optimisé est : (accuracy + précision) / 2.
    """

    # Définition du score custom
    def custom_score(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        return (acc + prec) / 2

    scorer = make_scorer(custom_score)

    # Grille de paramètres à tester
    param_grid = {
        "hidden_layer_sizes": [(20,10), (40,20)],  # moins de choix
        "activation": ["relu"],                    # un seul choix
        "solver": ["adam"],                        # un seul choix
        "max_iter": [500]                          # un seul choix
    }

    mlp = MLPClassifier(random_state=1)

    grid = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=1
    )

    grid.fit(Xtrain, Ytrain)

    print("===== Meilleurs paramètres trouvés =====")
    print(grid.best_params_)
    print(f"Score optimisé : {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_



# ============================================================
#  Création + sauvegarde d’un pipeline
# ============================================================


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


def create_pipeline(X, y, filename="credit_scoring_pipeline.pkl"):

    feature_union = FeatureUnion([
        ("scaler2", StandardScaler()),         # transformation supplémentaire
        ("pca", PCA(n_components=1))           # création d'une nouvelle variable
    ])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),          # normalisation initiale
        ("features", feature_union),           # ajout de transformations
        ("rfe", RFE(
            estimator=RandomForestClassifier(
                n_estimators=1000, random_state=1
            ),
            n_features_to_select=9
        )),
        ("mlp", MLPClassifier(
            activation='relu',
            hidden_layer_sizes=(10,),
            random_state=1
        ))
    ])

    pipeline.fit(X, y)

    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline sauvegardé dans {filename}")

    return pipeline

def run_classifiers_cv(X, y, clfs, n_splits=10):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    results = {}

    def metier_score(y_true, y_pred):
        return (accuracy_score(y_true, y_pred) +
                precision_score(y_true, y_pred)) / 2

    scorer_metier = make_scorer(metier_score)

    for name, clf in clfs.items():
        print(f"\n===== {name} =====")
        start = time.time()

        acc = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")

        if hasattr(clf, "predict_proba") or hasattr(clf, "decision_function"):
            auc = cross_val_score(clf, X, y, cv=kf, scoring="roc_auc")
        else:
            auc = np.full(n_splits, np.nan)

        metier = cross_val_score(clf, X, y, cv=kf, scoring=scorer_metier)

        exec_time = time.time() - start

        results[name] = {
            "accuracy_mean": acc.mean(),
            "accuracy_std": acc.std(),
            "auc_mean": auc.mean(),
            "auc_std": auc.std(),
            "metier_mean": metier.mean(),
            "metier_std": metier.std(),
            "exec_time": exec_time
        }

        print(f"Accuracy = {acc.mean():.3f} ± {acc.std():.3f}")
        print(f"AUC      = {auc.mean():.3f} ± {auc.std():.3f}")
        print(f"(Acc+Prec)/2 = {metier.mean():.3f} ± {metier.std():.3f}")
        print(f"Temps d'exécution : {exec_time:.2f} sec")

    # ==============================
    # Sélection du meilleur modèle selon l'AUC
    # ==============================
    best_model_name = max(results, key=lambda k: results[k]["auc_mean"])
    best_auc = results[best_model_name]["auc_mean"]

    print("\n==================== SÉLECTION FINALE ====================")
    print(f"Meilleur algorithme selon l'AUC : {best_model_name}")
    print(f"AUC moyenne = {best_auc:.3f}")

    best_model = clfs[best_model_name]

    return results


# ============================================================
# pipeline_generation_train_test_split
# ============================================================
def pipeline_generation_train_test_split(X, y, filename="pipeline_final11.pkl"):

    # 1) Séparation apprentissage / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    print("\n[1] Séparation train/test effectuée.")

    # 2) Apprentissage et évaluation sur données brutes
    models = {
        "CART": DecisionTreeClassifier(random_state=1),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(40,20), random_state=1)
    }

    print("\n[2] Résultats sur données brutes :")
    best_model_raw = run_classifiers_train_test(
        models, X_train, y_train, X_test, y_test, metric="precision"
    )

    # 3) Normalisation
    X_train_norm, X_test_norm, scaler = normalize_data(X_train, X_test)
    print("\n[3] Normalisation effectuée.")

    print("\n[4] Résultats sur données normalisées :")
    best_model_norm = run_classifiers_train_test(
        models, X_train_norm, y_train, X_test_norm, y_test, metric="precision"
    )

    # 4) Ajout des variables PCA
    X_train_ext, X_test_ext, pca = apply_pca_and_extend(X_train_norm, X_test_norm)
    print("\n[5] Variables PCA ajoutées.")

    print("\n[6] Résultats sur données normalisées + PCA :")
    best_model_ext = run_classifiers_train_test(
        models, X_train_ext, y_train, X_test_ext, y_test, metric="precision"
    )

    # ==== CHOIX STRATÉGIE OPTIMALE ====
    print("\n[7] Choix de la meilleure stratégie…")

    # On choisit automatiquement le meilleur score final
    strategies = {
        "brutes": best_model_raw,
        "normalisées": best_model_norm,
        "normalisées + PCA": best_model_ext
    }
    # On garde la dernière clé (= dernier best retourné)
    best_strategy = list(strategies.keys())[-1]
    print(f"→ Meilleure stratégie : {best_strategy}")

    # Pour la suite, on utilise les données étendues :
    Xtrain_used = X_train_ext
    Xtest_used  = X_test_ext

    # 5) Importance des variables
    print("\n[8] Importance des variables…")
    col_names = np.array([f"Var_{i}" for i in range(Xtrain_used.shape[1])])
    sorted_idx = variable_importance(Xtrain_used, y_train, col_names)

    # 6) Sélection du nombre optimal de variables
    print("\n[9] Sélection du nombre optimal de variables…")
    best_f, scores = select_nb_variables(
        Xtrain_used, Xtest_used, y_train, y_test, sorted_idx
    )

    # 7) Tuning MLP
    print("\n[10] Recherche des meilleurs paramètres MLP…")
    best_model, best_params, best_score = tune_mlp_hyperparameters(
        Xtrain_used[:, sorted_idx[:best_f]],
        y_train
    )

    print(f"Meilleurs paramètres MLP : {best_params}")

    # 8) Construction et sauvegarde du pipeline final
    print("\n[11] Création du pipeline final…")
    final_pipeline = create_pipeline(X, y, filename=filename)

    print(f"\nPipeline sauvegardé dans : {filename}")

    return final_pipeline




#Generation pipline general cv
def pipeline_generation_cv(X, y, clfs, filename="pipeline_final.pkl"):

    print("\n================ VALIDATION CROISÉE =================\n")

    # 1) Comparaison des algorithmes par CV
    cv_results = run_classifiers_cv(X, y, clfs)

    # 2) Choix du meilleur modèle selon l'AUC
    best_model_name = max(
        cv_results,
        key=lambda k: cv_results[k]["auc_mean"]
    )

    print(f"\n>>> Meilleur algorithme selon l'AUC : {best_model_name}")
    print(f"AUC moyenne = {cv_results[best_model_name]['auc_mean']:.3f}\n")

    # 3) Récupération du meilleur classifieur
    best_clf = clfs[best_model_name]

    # 4) Création du pipeline FINAL (normalisation + classifieur)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", best_clf)
    ])

    pipeline.fit(X, y)

    # 5) Sauvegarde
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline final sauvegardé dans : {filename}")

    return pipeline, best_model_name, cv_results

