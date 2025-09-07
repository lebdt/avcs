
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import itertools
# import statsmodels.api as sm
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

import warnings
warnings.filterwarnings("ignore")


def run_iris():
    df = pd.read_csv("../samples/iris.csv")


    result = None

    # 1. Preparação dos dados
    X = df.drop(columns='class')
    y = df['class']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    # 2. Todas as combinações possíveis de pares de variáveis
    variaveis = list(X.columns)
    combinacoes = list(itertools.combinations(variaveis, 2))

    # 3. Iterar por cada dupla
    novas_variaveis = []

    for var1, var2 in combinacoes:
        X_temp = X_train[[var1, var2]]
        X_val_temp = X_val[[var1, var2]]

        # a) Treinar árvore de decisão
        clf = DecisionTreeClassifier()
        clf.fit(X_temp, y_train)
        prob_pred = clf.predict_proba(X_temp)[:, 1]

        # b) Ajustar regressões lineares
        X_reg = sm.add_constant(X_temp)
        modelo_sem_interacao = sm.OLS(prob_pred, X_reg).fit()

        # Com interação
        X_temp['interacao'] = X_temp[var1] * X_temp[var2]
        X_reg2 = sm.add_constant(X_temp)
        modelo_com_interacao = sm.OLS(prob_pred, X_reg2).fit()

        # c) Calcular R² na validação
        X_val_temp['interacao'] = X_val_temp[var1] * X_val_temp[var2]
        X_val_reg = sm.add_constant(X_val_temp)

        y_pred_sem = modelo_sem_interacao.predict(sm.add_constant(X_val[[var1, var2]]))
        y_pred_com = modelo_com_interacao.predict(X_val_reg)

        r2_pre = r2_score(clf.predict_proba(X_val[[var1, var2]])[:, 1], y_pred_sem)
        r2_pos = r2_score(clf.predict_proba(X_val[[var1, var2]])[:, 1], y_pred_com)

        if (r2_pos - r2_pre) > 0.01:  # limite de melhoria, ajustável
            novas_variaveis.append((var1, var2, 'interacao', r2_pos - r2_pre))

    # Manter no máximo 31 novas features
    if len(novas_variaveis) > 31:
        novas_variaveis = sorted(novas_variaveis, key=lambda x: x[3], reverse=True)[:31]

    result = {}
    for i in range(1, len(novas_variaveis) + 1):
        result[str(i)] = {
            "var_one": novas_variaveis[i-1][0],
            "var_two": novas_variaveis[i-1][1],
            "operator": "*",
            "rsq": novas_variaveis[i-1][3],
        }


    print(novas_variaveis)
    for variavel_tup in novas_variaveis[::-1][0:3]:
        var1 = variavel_tup[0]
        var2 = variavel_tup[1]
        df[f'{var1}_x_{var2}'] = df[var1] * df[var2]



    preview_html = (
        df.sample(n=45)
        .reset_index(drop=True)
        .to_html(classes="table table-striped")
    )

    # col_type = str(df[column_name].dtype)
    #
    # col_stats = {
    #     "type": col_type,
    #     "missing": float(df[column_name].isna().sum()),
    #     "unique_values": float(df[column_name].nunique()),
    # }

    # if col_type.startswith(("int", "float")):
    #     col_values = df[column_name].astype(float)
    #
    #     col_stats.update(
    #         {
    #             "mean": col_values.mean(),
    #             "median": col_values.median(),
    #             "std": col_values.std(),
    #             "min": col_values.min(),
    #             "max": col_values.max(),
    #         }
    #     )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    stats = None


    q1 = f"""
    This is the result of the 'variable pairing test' for my dataset.

    {result}

    You're the specialist. Tell me the best interpretation for those results.
    """

    # resp = prompt(q1)
    # prompt_result = (
    #     md.markdown(resp).replace("<h1>", "<h3>").replace("</h1>", "</h3>")
    # )

    prompt_result = "Chat Response"

    return {
        "type": "pairs_tested",
        "result": result,
        "prompt_response": prompt_result,
        "preview": preview_html,
        # "column_name": column_name,
        # "column_stats": col_stats,
        "stats": stats,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "success": True,
        "df": df,
    }


# print(run_iris()["df"].columns)


import pandas as pd

a = pd.DataFrame({'class': [0, 2], 'class2': ['a', 'b'], 'col1': [0.1, 0.2]})



elem = 'class'
n_col = 'new_col'

import keyword
klist = keyword.kwlist

formula = f"{elem} + 0"

try:
    # a = a.eval(formula)
    print(klist)
except SyntaxError as e:
    is_kw = [i for i in a.columns if i in klist]


    for kwc in is_kw:
        a[kwc + '__00remove00__'] = a[kwc]
        formula = formula.replace(kwc, kwc + '__00__00__')

    a = a.eval(formula)
    for rmcol in a.columns:
        if '__00remove00__' in rmcol:
            a.drop(rmcol)


    elem2 = elem + '_00_'
    a[elem2] = a[elem]
    a[n_col] = a.eval(f'`{elem2}` + 213')
    a = a.drop(elem2, axis=1)
    print(e)
    print(a)


