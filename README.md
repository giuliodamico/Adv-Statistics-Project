1. EDA
2. Fittare diversi modelli di linear regression (OSL, lasso, etc) e vedere quale è il migliore
3. Fare test statistici sul modello migliore (confidence interval, etc)
4. FIttare un modello random forest con crossvalidation per predirre una delle variabili categoriche
5. interpretare il modello di random forest

1) EDA
Obiettivi: capire target, scala e distribuzione, outlier, missingness, collinearità, leakage, e feature categoriche/ordinali per encoding adeguato.
Suggerimenti:
Profilazione: distribuzioni, boxplot per outlier, heatmap di correlazioni, VIF iniziale per sospetta multicollinearità.
Dati stradali: controlla coerenza temporale/spaziale, categorie ISTAT/ACI, e codifiche standardizzate; rivedi imputazioni con buon senso (es. orario/incrocio mancanti).
  ​
2) Regressione (target continuo): OLS, Ridge, Lasso, Elastic Net
    Preprocessing:
        Split stratificato sul range del target o k-fold nested; scaling per modelli penalizzati.
​One-hot per categoriche, gestione missing (imputer + indicatore di missing se informativo).
   ​Selezione modello:
Confronta OLS, Ridge, Lasso, Elastic Net con k-fold (idealmente nested CV per l’hyper-tuning). Metriche: RMSE/MAE come primarie, R² solo come complemento; usa errori relativi se le unità contano meno.​
Controlla collinearità: VIF post-selezione o robustezza delle stime in Ridge/EN se molte feature correlate.  ​
Diagnostica sul candidato migliore:
    Residui: omoschedasticità (residui vs pred), normalità residui per inference (QQ plot), autocorrelazione se dati temporali.
Influential points: leverage e Cook’s distance; verifica stabilità delle performance togliendo punti estremi.
3) Inference e test sul modello migliore
    Intervalli di confidenza:
        Per OLS classico: IC su coefficienti con varianza robusta (HC3) se c’è eteroschedasticità.
Per Lasso/EN: inference standard non è diretta; valuta stability selection, debiased Lasso, o bootstrap per IC/SE “empiriche”.
    Test:
    Significatività globale (F-test), test sui singoli coefficienti, test di White/Breusch-Pagan per eteroschedasticità, Durbin–Watson su serie temporali.
Calibrazione predittiva:
    Curve di affidabilità predittiva via binning su ŷ per valutare bias sistematici su sotto-range del target.
4) Random Forest per classificazione (variabile categorica)
    Setup:
        Scelta variabile target bilanciata o applica class_weight/balanced subsampling; attenzione a leakage temporale/spaziale.
Pipeline: imputazione, encoding categoriche (one-hot va bene; RF gestisce non linearità).
    ​
Cross-validation:
    k-fold stratificata; se dati spaziali/temporali, usa blocchi (GroupKFold/TimeSeriesSplit) per evitare overfitting di prossimità.
Hyper-tuning: n_estimators, max_depth, max_features, min_samples_split/leaf; ottimizza sulla metrica rilevante.
Metriche:
    Classi bilanciate: accuracy, macro-F1; sbilanciate: ROC-AUC, PR-AUC, balanced accuracy; reporta confusion matrix e curve.
Calibrazione: isotonic o Platt se probabilità utilizzate in decisioni o ranking.
5) Interpretazione del modello RF
    Importanza:
        Permutation importance su CV per robustezza; evita reliance esclusiva su Gini importance.
Effetti:
    PDP e ICE per feature chiave; SHAP TreeExplainer per contributi locali e globali, con attenzione all’interpretabilità in presenza di feature correlate.​
Stabilità:
    Verifica la stabilità dei rank di importanza su fold e su bootstrap; segnala feature instabili.
Migliorie consigliate alla pipeline
    Validazione annidata: usa nested CV per confronti corretti tra modelli regressivi e per l’RF, evitando optimistic bias.​
Leakage control: separa accuratamente ogni operazione dipendente dai dati (imputazione/encoding/scaling) dentro pipeline fitted SOLO sul train per ciascun fold.
Reporting replicabile: fissa random_state, salva environment (requirements.txt), logga split, hyperparam, seed, e versiona i notebook in step chiari.
Analisi dominio incidenti:
    Integra indicatori standardizzati e definizioni ICD/ISTAT quando presenti, e documenta trasformazioni delle variabili coerenti con la pratica ufficiale.
Se spazio/tempo rilevanti: feature engineering su orario/giorno/meteo, densità traffico o proxy, clustering per “hotspots”, e valutazioni per area.
Struttura pratica dei notebook
    00_eda.ipynb: profilazione, cleaning, leakage check, split strategy motivata.
01_regression_models.ipynb: OLS/Ridge/Lasso/EN con nested CV, confronti e diagnostica residui sul best.
02_regression_inference.ipynb: IC, test, robust SE o bootstrap.
03_rf_classification.ipynb: tuning con CV, metriche, calibrazione, confusion matrix, curve ROC/PR.
04_rf_interpretability.ipynb: permutation importance, PDP/ICE, SHAP, analisi stabilità.
