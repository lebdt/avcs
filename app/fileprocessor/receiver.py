import json
import pandas as pd
import os

from pandas.core.arrays import categorical
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from channels.db import database_sync_to_async
from .models import CSVFile
import tabulate as tab
import markdown as md
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    confusion_matrix,
)
from sklearn import tree


def prompt(input_text, key=settings.OPENAI_API_KEY):
    from openai import OpenAI

    premise = """
    You are AVCS, an Automated Variable Creation System designed to help users analyze datasets and suggest meaningful new variables interactively. Your primary role is to interpret outputs from statistical techniques such as PCA (Principal Component Analysis) and MDA (Multiple Discriminant Analysis) and guide users in selecting, transforming, or engineering variables to enhance their models.

    Guidelines for Effective Assistance:
    1. Provide Data-Driven Insights:

    Explain patterns found in PCA and MDA outputs clearly and practically.
    Suggest potential new variables based on factor loadings, eigenvalues, discriminant functions, and variance explained.
    Guide users through dimensionality reduction and feature selection decisions.

    2. Ensure Clarity and Accessibility:

    Use straightforward explanations, avoiding unnecessary jargon.
    Adapt explanations based on the user’s experience level (beginner, intermediate, advanced).
    Offer real-world examples where similar techniques have been applied.

    3. Foster an Iterative Approach:

    Allow users to refine variable suggestions step by step.
    Highlight the trade-offs between complexity, interpretability, and predictive power.
    Provide guidance on testing and validating new variables within models.
    Example Interaction:
    User: "I ran PCA on my dataset, and the first two components explain 75% of the variance. How can I use this for variable creation?"

    AVCS:
    "The fact that the first two components explain 75% of the variance suggests that much of the dataset's information is concentrated in a lower-dimensional space. You have several options:

    Feature Compression: Create two new variables (PC1 and PC2) and replace multiple correlated features with them.
    Variable Selection: Identify the original features that contribute the most to PC1 and PC2 based on their loadings, and prioritize them in modeling.
    Interaction Features: If PC1 captures 'size' and PC2 captures 'shape,' you can create new variables that emphasize these dimensions.

    4. Straight to the point:
    Your answers MUST be direct. DO NOT reply like "Based on what you asked..." or "Based on what you told me..."


    IMPORTANT: Always wrap column names around backticks `[COLUMN_NAME]` for readability


    Suggestions are good but be concise.

    Now, the resquest:\n


    """

    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        temperature=0.05,
        messages=[
            {
                "role": "user",
                "content": premise + input_text,
            }
        ],
        model="gpt-4o",
    )
    parsed_response = response.choices[0].message.content


    return parsed_response


def get_stats(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [i for i in df.columns.tolist() if i not in numeric_cols]
    means = {col: df[col].mean() for col in numeric_cols}
    std_dev = {col: df[col].std() for col in numeric_cols}

    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "means": means,
        "stddev": std_dev,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    return stats


class CSVConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data_json = json.loads(text_data)
        action = data_json.get("action")

        if action == "process_csv":
            file_id = data_json.get("file_id")
            if file_id:
                results = await self.process_csv(file_id)
                await self.send(text_data=json.dumps(results))

        elif action == "query_data":
            file_id = data_json.get("file_id")
            query = data_json.get("query")
            if file_id and query:
                results = await self.execute_query(file_id, query)
                await self.send(text_data=json.dumps(results))

        elif action == "test_pairs":
            file_id = data_json.get("file_id")
            response = data_json.get("response")
            if file_id:
                results = await self.execute_pairs_test(file_id, response)
                await self.send(text_data=json.dumps(results))

        elif action == "analyze_data":
            file_id = data_json.get("file_id")
            analysis_type = data_json.get("analysis_type")
            predictors = data_json.get("predictors", [])
            response = data_json.get("response")

            if file_id and analysis_type:
                results = await self.analyze_data(
                    file_id, analysis_type, predictors, response
                )
                await self.send(text_data=json.dumps(results))

        elif action == "modify-dataset":
            file_id = data_json.get("file_id")
            results = None
            results = await self.send(text_data=json.dumps(results))

        elif action == "add_column":
            file_id = data_json.get("file_id")
            column_name = data_json.get("column_name")
            formula = data_json.get("formula")
            if file_id and column_name and formula:
                results = await self.add_column(file_id, column_name, formula)
                await self.send(text_data=json.dumps(results))

        elif action == "modify_column":
            file_id = data_json.get("file_id")
            column_name = data_json.get("column_name")
            formula = data_json.get("formula")
            if file_id and column_name and formula:
                results = await self.modify_column(file_id, column_name, formula)
                await self.send(text_data=json.dumps(results))

        elif action == "delete_column":
            file_id = data_json.get("file_id")
            column_name = data_json.get("column_name")
            if file_id and column_name:
                results = await self.delete_column(file_id, column_name)
                await self.send(text_data=json.dumps(results))

    @database_sync_to_async
    def process_csv(self, file_id):
        try:
            csv_file = CSVFile.objects.get(pk=file_id)
            df = pd.read_csv(csv_file.file.path)

            stats = get_stats(df)

            preview_html = (
                df.sample(n=45)
                .reset_index(drop=True)
                .to_html(classes="table table-striped")
            )

            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df.to_pickle(pickle_path)

            sample_rows = tab.tabulate(
                df.iloc[:15], showindex=False, tablefmt="psql", headers="keys"
            )

            sample_dtypes = df.dtypes

            q1 = f"""

            This is a sample of my dataset and its types
            {sample_rows}
            {sample_dtypes}

            Tell me what column or columns could be good response variable(s) AND Give me a brief and best possible explanation of this data

            Explain each variable evaluated in bullet points and make explicit their type.


            E.g.

            **General Explanation:**

            [GENERAL EXPLANATION]

            **Best potential response variable**: `RESPONSE_VAR`

            **Other possible response variables**: `OTHER_RESPONSE_VARS` (IMPORTANT: This line is optional)

             - `VAR_1`, *numeric*: [EXPLANATION]
            """
            resp = prompt(q1)
            result = md.markdown(resp).replace("<h1>", "<h3>").replace("</h1>", "</h3>")

            import re

            def get_seq(compare, origin):
                seq_list = []
                subseq = {}
                for i in range(len(compare)):
                    subseq[i] = compare[i:]
                    subseq[i + len(compare)] = compare[: len(compare) - i]
                    for m in range(i, len(compare)):
                        if len(compare) - m > i:
                            subseq[i + 2 * len(compare) + m] = compare[
                                i : len(compare) - m
                            ]
                    seq_list.append(subseq)

                isin_seq = []
                for j in subseq.keys():
                    if subseq[j] in origin:
                        isin_seq.append((subseq[j], len(subseq[j])))
                res = set(isin_seq)
                return res

            def get_chars(compare, origin):
                s_length = compare if len(compare) < len(origin) else origin
                b_length = compare if len(compare) > len(origin) else origin
                res = [c for c in s_length if c in b_length]
                return len(res)

            def ff(compare, origin, case_insensitive=True):
                if case_insensitive:
                    compare = compare.lower()
                    origin = origin.lower()
                chars = get_chars(compare, origin)
                seqs = get_seq(compare, origin)
                if chars == 0:
                    rank = 0
                    return rank
                elif seqs == set():
                    max_seq_len = 1e-10
                else:
                    seq_len = set([s[1] for s in seqs])
                    max_seq_len = max(seq_len)
                rel_chars = chars / len(origin)
                rank = 56 / (1 / rel_chars + 55 / max_seq_len)
                return rank

            def best_match(compare, string_list):
                c_tups = [(c, ff(c, compare)) for c in string_list]
                print(c_tups)
                default_response = sorted(c_tups, key=lambda x: x[1], reverse=True)[0][0]
                return default_response

            re_search = re.search("(.*Best potential response.*):(.*)\n", resp)

            response_var = None
            if re_search:
                str_response = (
                    re_search.group(2).strip()
                )
                print(str_response)

                response_var = best_match(str_response, df.columns.tolist())
                print("resp. var ", response_var)


            return {
                "type": "csv_processed",
                "stats": stats,
                "preview": preview_html,
                "file_id": file_id,
                "overview": result,
                "default_response": response_var,
                "success": True,
            }
        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def execute_pairs_test(self, file_id, response):
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            import pandas as pd
            import numpy as np
            import itertools
            import statsmodels.api as sm
            from sklearn import datasets
            from sklearn.datasets import load_breast_cancer

            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            result = None

            # 1. Preparação dos dados
            X = df.drop(columns=response).select_dtypes(include=['number'])
            y = df[response]
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
                X_temp["interacao"] = X_temp[var1] * X_temp[var2]
                X_reg2 = sm.add_constant(X_temp)
                modelo_com_interacao = sm.OLS(prob_pred, X_reg2).fit()

                # c) Calcular R² na validação
                X_val_temp["interacao"] = X_val_temp[var1] * X_val_temp[var2]
                X_val_reg = sm.add_constant(X_val_temp)

                y_pred_sem = modelo_sem_interacao.predict(
                    sm.add_constant(X_val[[var1, var2]])
                )
                y_pred_com = modelo_com_interacao.predict(X_val_reg)

                r2_pre = r2_score(
                    clf.predict_proba(X_val[[var1, var2]])[:, 1], y_pred_sem
                )
                r2_pos = r2_score(
                    clf.predict_proba(X_val[[var1, var2]])[:, 1], y_pred_com
                )

                if (r2_pos - r2_pre) > 0.01:  # limite de melhoria, ajustável
                    novas_variaveis.append((var1, var2, "interacao", r2_pos - r2_pre))

            # Manter no máximo 31 novas features
            if len(novas_variaveis) > 31:
                novas_variaveis = sorted(
                    novas_variaveis, key=lambda x: x[3], reverse=True
                )[:31]

            result = {}
            for i in range(1, len(novas_variaveis) + 1):
                result[str(i)] = {
                    "var_one": novas_variaveis[i - 1][0],
                    "var_two": novas_variaveis[i - 1][1],
                    "operator": "*",
                    "rsq": novas_variaveis[i - 1][3],
                }

            for variavel_tup in novas_variaveis[::-1][0:3]:
                var1 = variavel_tup[0]
                var2 = variavel_tup[1]
                df[f"{var1}_x_{var2}"] = df[var1] * df[var2]

            df.to_pickle(pickle_path)

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
            stats = get_stats(df)

            print(result)

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
            }

        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def execute_query(self, file_id, query):
        try:
            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            result = None

            if query.startswith("filter:"):
                filter_expr = query.replace("filter:", "")
                possible_op = [">=", ">", "!=", "=", "<=", "<"]
                op = [o for o in possible_op if o in filter_expr][0]
                op_val = filter_expr.split(op, 1)
                col = op_val[0].strip()
                val = op_val[1].strip()

                if op == ">":
                    result = df[df[col] > float(val)]
                elif op == "<":
                    result = df[df[col] < float(val)]
                elif op == "=":
                    result = df[df[col] == val]
                elif op == "!=":
                    result = df[df[col] != float(val)]
                elif op == "<=":
                    result = df[df[col] <= float(val)]
                elif op == ">=":
                    result = df[df[col] >= float(val)]

            elif query.startswith("groupby:"):

                try:
                    groupby_col = query.replace("groupby:", "").split(';')[0].split(',')[0].strip()
                    groupby_agg = query.replace("groupby:", "").split(';')[0].split(',')[1].strip()
                    groupby_agg_cols = [i.strip() for i in query.replace("groupby:", "").split(';')[1].split(',')]
                except:
                    groupby_col = query.replace("groupby:", "").split(',')[0].strip()
                    groupby_agg = query.replace("groupby:", "").split(',')[1].strip()
                    groupby_agg_cols = None

                numerical_cols = [i for i in df.select_dtypes(include=["number"]).columns.tolist() if i != groupby_col]

                if groupby_agg_cols:
                    if groupby_agg in ['mean', 'avg']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].mean().reset_index()
                    elif groupby_agg in ['sum']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].sum().reset_index()
                    elif groupby_agg in ['std', 'stddev']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].stddev().reset_index()
                    elif groupby_agg in ['max']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].max().reset_index()
                    elif groupby_agg in ['min']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].min().reset_index()
                    elif groupby_agg in ['count']:
                        result = df.groupby(groupby_col)[groupby_agg_cols].count().reset_index()
                else:
                    if groupby_agg in ['mean', 'avg']:
                        result = df.groupby(groupby_col)[numerical_cols].mean().reset_index()
                    elif groupby_agg in ['sum']:
                        result = df.groupby(groupby_col)[numerical_cols].sum().reset_index()
                    elif groupby_agg in ['std', 'stddev']:
                        result = df.groupby(groupby_col)[numerical_cols].stddev().reset_index()
                    elif groupby_agg in ['max']:
                        result = df.groupby(groupby_col)[numerical_cols].max().reset_index()
                    elif groupby_agg in ['min']:
                        result = df.groupby(groupby_col)[numerical_cols].min().reset_index()
                    elif groupby_agg in ['count']:
                        result = df.groupby(groupby_col).count().reset_index()

            elif query.startswith("sort:"):
                sort_params = query.replace("sort:", "").split(",")
                col = sort_params[0]
                asc = True if len(sort_params) < 2 or sort_params[1] == "asc" else False
                result = df.sort_values(by=col, ascending=asc)

            elif query == "describe":
                result = df.describe()

            elif query == "intro":
                q1 = f"""

                Tell me what you are.
                """

                resp = prompt(q1)
                result = (
                    md.markdown(resp).replace("<h1>", "<h3>").replace("</h1>", "</h3>")
                )
                return {"type": "query_result", "result": result, "success": True}

            elif query == "overview":
                sample_rows = tab.tabulate(
                    df.iloc[:15], showindex=False, tablefmt="psql", headers="keys"
                )

                sample_dtypes = df.dtypes

                q1 = f"""

                This is a sample of my dataset and its types
                {sample_rows}
                {sample_dtypes}

                Tell me what column or columns could be good response variable(s) AND Give me a brief and best possible explanation of this data

                Explain each variable evaluated in bullet points and make explicit their type.


                E.g.

                **General Explanation**

                [GENERAL EXPLANATION]


                **Best potential response variable(s)**: `RESPONSE_VAR(S)`

                 - `VAR_1`, *numeric*: [EXPLANATION]
                """
                resp = prompt(q1)
                result = md.markdown(resp).replace("<h1>", "<h3>").replace("</h1>", "</h3>")

                import re

                def get_seq(compare, origin):
                    seq_list = []
                    subseq = {}
                    for i in range(len(compare)):
                        subseq[i] = compare[i:]
                        subseq[i + len(compare)] = compare[: len(compare) - i]
                        for m in range(i, len(compare)):
                            if len(compare) - m > i:
                                subseq[i + 2 * len(compare) + m] = compare[
                                    i : len(compare) - m
                                ]
                        seq_list.append(subseq)

                    isin_seq = []
                    for j in subseq.keys():
                        if subseq[j] in origin:
                            isin_seq.append((subseq[j], len(subseq[j])))
                    res = set(isin_seq)
                    return res

                def get_chars(compare, origin):
                    s_length = compare if len(compare) < len(origin) else origin
                    b_length = compare if len(compare) > len(origin) else origin
                    res = [c for c in s_length if c in b_length]
                    return len(res)

                def ff(compare, origin, case_insensitive=True):
                    if case_insensitive:
                        compare = compare.lower()
                        origin = origin.lower()
                    chars = get_chars(compare, origin)
                    seqs = get_seq(compare, origin)
                    if chars == 0:
                        rank = 0
                        return rank
                    elif seqs == set():
                        max_seq_len = 1e-10
                    else:
                        seq_len = set([s[1] for s in seqs])
                        max_seq_len = max(seq_len)
                    rel_chars = chars / len(origin)
                    rank = 56 / (1 / rel_chars + 55 / max_seq_len)
                    return rank

                def best_match(compare, string_list):
                    c_tups = [(c, ff(c, compare)) for c in string_list]
                    default_response = sorted(c_tups, key=lambda x: x[1], reverse=True)[0][0]
                    return default_response

                re_search = re.search("(.*Best potential response.*):(.*)\n", resp)

                response_var = None
                if re_search:
                    str_response = (
                        re_search.group(2).strip()
                    )
                    print(str_response)

                    response_var = best_match(str_response, df.columns.tolist())
                    print("resp. var ", response_var)


                return {
                    "type": "query_result",
                    "result": result,
                    "default_response": response_var,
                    "success": True,
                }

            else:
                return {
                    "type": "query_result",
                    "result": "Unsupported query type",
                    "success": False,
                }

            if result is not None:
                result_html = result.head(100).to_html(classes="table table-striped")
                return {
                    "type": "query_result",
                    "rows": result.shape[0],
                    "result": result_html,
                    "success": True,
                }

        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def analyze_data(self, file_id, analysis_type, predictors, response=None):
        import numpy as np

        try:
            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            for col in predictors:
                if col not in df.columns:
                    return {
                        "type": "error",
                        "message": f"Column '{col}' not found in dataset",
                        "success": False,
                    }

            if response and response not in df.columns:
                return {
                    "type": "error",
                    "message": f"Response column '{response}' not found in dataset",
                    "success": False,
                }

            if predictors:
                X = df[predictors]
                X = X.select_dtypes(include=["number"])
                if X.shape[1] == 0:
                    return {
                        "type": "error",
                        "message": "No numeric predictor columns selected",
                        "success": False,
                    }
            else:
                X = df.select_dtypes(include=["number"])

            result = {}
            plots = {}

            if analysis_type == "summary":
                result["summary"] = X.describe().to_html(classes="table table-striped")
                result["correlation"] = X.corr().to_html(classes="table table-striped")
                result["missing"] = pd.DataFrame(
                    {
                        "Missing Values": X.isnull().sum(),
                        "Percentage": X.isnull().sum() / len(X) * 100,
                    }
                ).to_html(classes="table table-striped")

            elif analysis_type == "pca":
                if X.shape[1] < 2:
                    return {
                        "type": "error",
                        "message": "PCA requires at least 2 numeric columns",
                        "success": False,
                    }

                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA

                X_scaled = StandardScaler().fit_transform(X)

                pca = PCA()
                pca_result = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame(
                    data=pca_result[:, 0 : min(5, X.shape[1])],
                    columns=[f"PC{i+1}" for i in range(min(5, X.shape[1]))],
                )

                explained_variance = pd.DataFrame(
                    {
                        "Component": [
                            f"PC{i+1}"
                            for i in range(len(pca.explained_variance_ratio_))
                        ],
                        "Explained Variance (%)": pca.explained_variance_ratio_ * 100,
                        "Cumulative Variance (%)": np.cumsum(
                            pca.explained_variance_ratio_
                        )
                        * 100,
                    }
                )

                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
                    index=X.columns,
                )

                result["pca_preview"] = pca_df.head(10).to_html(
                    classes="table table-striped"
                )
                result["explained_variance"] = explained_variance.to_html(
                    classes="table table-striped"
                )
                result["loadings"] = loadings.to_html(classes="table table-striped")

                scree_html = f"""
                <svg viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg">
                    <line x1="50" y1="250" x2="450" y2="250" stroke="black" />
                    <line x1="50" y1="250" x2="50" y2="50" stroke="black" />
                    <text x="250" y="290" text-anchor="middle">Principal Component</text>
                    <text x="20" y="150" text-anchor="middle" transform="rotate(270,20,150)">Variance Explained (%)</text>
                """

                width = 400 / len(pca.explained_variance_ratio_)
                max_height = 200
                for i, var in enumerate(pca.explained_variance_ratio_):
                    height = var * 100 * max_height / 100
                    x = 50 + i * width
                    y = 250 - height
                    scree_html += f"""
                    <rect x="{x}" y="{y}" width="{width*0.8}" height="{height}" fill="steelblue" />
                    <text x="{x + width*0.4}" y="270" text-anchor="middle">{i+1}</text>
                    <text x="{x + width*0.4}" y="{y-5}" text-anchor="middle">{(var*100):.1f}%</text>
                    """

                scree_html += "</svg>"
                plots["scree_plot"] = scree_html

            elif analysis_type == "correlation":
                corr_matrix = X.corr()

                n_cols = len(corr_matrix.columns)
                cell_size = min(400 // n_cols, 40)
                width = cell_size * n_cols + 100
                height = cell_size * n_cols + 100

                heatmap_html = f"""
                <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
                """

                for i, col in enumerate(corr_matrix.columns):
                    heatmap_html += f"""
                    <text x="{i * cell_size + 50 + cell_size/2}" y="20" 
                          text-anchor="end" transform="rotate(-45,{i * cell_size + 50 + cell_size/2},20)"
                          font-size="{min(cell_size*0.8, 6)}">{col}</text>
                    """

                for i, col in enumerate(corr_matrix.index):
                    heatmap_html += f"""
                    <text x="40" y="{i * cell_size + 50 + cell_size/2}" 
                          text-anchor="end" font-size="{min(cell_size*0.8, 6)}">{col}</text>
                    """

                for i, row in enumerate(corr_matrix.index):
                    for j, col in enumerate(corr_matrix.columns):
                        value = corr_matrix.loc[row, col]

                        if value > 0:
                            intensity = int(min(value * 255, 255))
                            color = f"rgb({255-intensity},{255},{255-intensity})"
                        else:
                            intensity = int(min(abs(value) * 255, 255))
                            color = f"rgb({255},{255-intensity},{255-intensity})"

                        heatmap_html += f"""
                        <rect x="{j * cell_size + 50}" y="{i * cell_size + 50}" 
                              width="{cell_size}" height="{cell_size}" 
                              fill="{color}" stroke="white" />
                        <text x="{j * cell_size + 50 + cell_size/2}" y="{i * cell_size + 50 + cell_size/2 + 5}" 
                              text-anchor="middle" font-size="{min(cell_size*0.6, 10)}">{value:.2f}</text>
                        """

                heatmap_html += "</svg>"
                plots["correlation_heatmap"] = heatmap_html
                result["correlation"] = corr_matrix.to_html(
                    classes="table table-striped"
                )

            elif analysis_type == "regression" and response:
                if df[response].dtype.kind not in "if":
                    return {
                        "type": "error",
                        "message": f"Response variable '{response}' must be numeric for regression",
                        "success": False,
                    }

                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score

                y = df[response]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                coef_df = pd.DataFrame(
                    {"Feature": X.columns, "Coefficient": model.coef_}
                )
                coef_df = coef_df.sort_values("Coefficient", ascending=False)

                result[
                    "model_summary"
                ] = f"""
                <div class="card mb-3">
                    <div class="card-header">Regression Model: Predicting {response}</div>
                    <div class="card-body">
                        <p><strong>Mean Squared Error:</strong> {mse:.4f}</p>
                        <p><strong>R² Score:</strong> {r2:.4f}</p>
                        <p><strong>Intercept:</strong> {model.intercept_:.4f}</p>
                    </div>
                </div>
                """
                result["coefficients"] = coef_df.to_html(
                    classes="table table-striped", index=False
                )

                scatter_html = f"""
                <svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
                    <line x1="50" y1="350" x2="450" y2="350" stroke="black" />
                    <line x1="50" y1="350" x2="50" y2="50" stroke="black" />
                    <text x="250" y="390" text-anchor="middle">Actual</text>
                    <text x="20" y="200" text-anchor="middle" transform="rotate(270,20,200)">Predicted</text>
                """

                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                range_val = max_val - min_val

                scatter_html += f"""
                <line x1="50" y1="350" x2="450" y2="50" stroke="blue" stroke-dasharray="5,5" />
                """

                for actual, predicted in zip(y_test, y_pred):
                    x = 50 + (actual - min_val) * 400 / range_val
                    y = 350 - (predicted - min_val) * 300 / range_val
                    scatter_html += f"""
                    <circle cx="{x}" cy="{y}" r="3" fill="red" />
                    """

                scatter_html += "</svg>"
                plots["regression_scatter"] = scatter_html

            elif analysis_type == "clustering":
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                X_scaled = StandardScaler().fit_transform(X)

                max_clusters = min(10, X.shape[0] // 5)
                wcss = []
                for i in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)

                k = 2
                if len(wcss) > 2:
                    diffs = np.diff(wcss)
                    diffs_of_diffs = np.diff(diffs)
                    if len(diffs_of_diffs) > 0:
                        k = np.argmax(diffs_of_diffs) + 2

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                cluster_df = X.copy()
                cluster_df["Cluster"] = clusters

                centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
                centers_df.index.name = "Cluster"

                cluster_counts = pd.Series(clusters).value_counts().sort_index()

                result[
                    "kmeans_summary"
                ] = f"""
                <div class="card mb-3">
                    <div class="card-header">K-Means Clustering</div>
                    <div class="card-body">
                        <p><strong>Number of Clusters:</strong> {k}</p>
                        <p><strong>Cluster Sizes:</strong></p>
                        <ul>
                            {''.join([f'<li>Cluster {i}: {count} samples</li>' for i, count in enumerate(cluster_counts)])}
                        </ul>
                    </div>
                </div>
                """
                result["cluster_centers"] = centers_df.to_html(
                    classes="table table-striped"
                )
                result["clustered_data"] = cluster_df.head(10).to_html(
                    classes="table table-striped"
                )

                elbow_html = f"""
                <svg viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg">
                    <line x1="50" y1="250" x2="450" y2="250" stroke="black" />
                    <line x1="50" y1="250" x2="50" y2="50" stroke="black" />
                    <text x="250" y="290" text-anchor="middle">Number of Clusters</text>
                    <text x="20" y="150" text-anchor="middle" transform="rotate(270,20,150)">WCSS</text>
                """

                if len(wcss) > 1:
                    points = []
                    max_wcss = max(wcss)
                    for i, w in enumerate(wcss):
                        x = 50 + i * (400 / (len(wcss) - 1))
                        y = 250 - (w / max_wcss * 200)
                        points.append(f"{x},{y}")

                    elbow_html += f"""
                    <polyline points="{' '.join(points)}" stroke="blue" fill="none" />
                    """

                    for i, w in enumerate(wcss):
                        x = 50 + i * (400 / (len(wcss) - 1))
                        y = 250 - (w / max_wcss * 200)
                        elbow_html += f"""
                        <circle cx="{x}" cy="{y}" r="4" fill="blue" />
                        <text x="{x}" y="{y-10}" text-anchor="middle">{i+1}</text>
                        """

                    x_k = 50 + (k - 1) * (400 / (len(wcss) - 1))
                    y_k = 250 - (wcss[k - 1] / max_wcss * 200)
                    elbow_html += f"""
                    <circle cx="{x_k}" cy="{y_k}" r="6" fill="red" stroke="black" />
                    """

                elbow_html += "</svg>"
                plots["elbow_curve"] = elbow_html

            elif analysis_type == "decisiontrees":
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_squared_error, r2_score
                import io, base64

                all_columns = predictors + [response]
                missing_columns = [col for col in all_columns if col not in df.columns]

                if missing_columns:
                    return {
                        "type": "error",
                        "message": f"Columns not found: {', '.join(missing_columns)}",
                        "success": False,
                    }

                X = df[predictors].select_dtypes(include=["number"]).copy()

                dropped_features = [col for col in predictors if col not in X.columns]
                if dropped_features:
                    warning_message = (
                        f"Dropped non-numeric features: {', '.join(dropped_features)}"
                    )
                else:
                    warning_message = None

                if X.empty or len(X.columns) == 0:
                    return {
                        "type": "error",
                        "message": "No numeric features available for analysis",
                        "success": False,
                    }

                X = X.fillna(X.mean())

                y = df[response]
                if pd.api.types.is_numeric_dtype(y):
                    is_classification = len(y.unique()) < 10
                else:
                    y = pd.factorize(y)[0]
                    is_classification = True

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                result = {}
                plots = {}

                if is_classification:
                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    result = {
                        "accuracy": float(accuracy),
                        "confusion_matrix": cm.tolist(),
                        "feature_importance": dict(
                            zip(X.columns, model.feature_importances_.tolist())
                        ),
                        "model_type": "classification_tree",
                    }

                    plt.figure(figsize=(12, 8))
                    tree.plot_tree(
                        model,
                        filled=True,
                        feature_names=X.columns,
                        proportion=True,
                        rounded=True,
                    )
                    plt.title(f"Decision Tree for '{response}' Classification")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    tree_img = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()

                    plt.figure(figsize=(10, 6))
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    importances = importances.sort_values(ascending=False)
                    importances.plot(kind="bar")
                    plt.title("Feature Importance")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    importance_img = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()

                    plots = {"tree": tree_img, "importance": importance_img}

                else:
                    model = DecisionTreeRegressor(max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    result = {
                        "r2_score": float(r2),
                        "mean_squared_error": float(mse),
                        "feature_importance": dict(
                            zip(X.columns, model.feature_importances_.tolist())
                        ),
                        "model_type": "regression_tree",
                    }

                    plt.figure(figsize=(12, 8))
                    tree.plot_tree(
                        model,
                        filled=True,
                        feature_names=X.columns,
                        proportion=True,
                        rounded=True,
                    )
                    plt.title(f"Decision Tree for {response} Regression")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    tree_img = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()

                    plt.figure(figsize=(10, 6))
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    importances = importances.sort_values(ascending=False)
                    importances.plot(kind="bar")
                    plt.title("Feature Importance")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    importance_img = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()

                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred)
                    plt.plot(
                        [y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()],
                        "k--",
                    )
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title("Actual vs Predicted Values")

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    pred_img = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()

                    plots = {
                        "tree": tree_img,
                        "importance": importance_img,
                        "predictions": pred_img,
                    }

            q1 = f"""
            This is the result of the '{analysis_type}' for my dataset.

            {result}

            You're the specialist. Tell me the best interpretation for those results. Remember that our job here is to combine variables.
            """

            resp = prompt(q1)
            prompt_result = (
                md.markdown(resp).replace("<h1>", "<h3>").replace("</h1>", "</h3>")
            )

            return {
                "type": "analysis_result",
                "analysis_type": analysis_type,
                "result": result,
                "plots": plots,
                "prompt_response": prompt_result,
                "success": True,
            }

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def add_column(self, file_id, column_name, formula):
        try:
            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            if column_name in df.columns:
                return {
                    "type": "error",
                    "message": f"Column <b>{column_name}</b> already exists.",
                    "success": False,
                }

            try:
                if formula.startswith("="):
                    formula = formula[1:]
                if "@" in formula:
                    return {
                        "type": "error",
                        "message": f"Formula error: Invalid character '@'",
                        "success": False,
                    }

                try:
                    result = df.eval(formula)
                except SyntaxError as e:
                    import keyword

                    kwlist = keyword.kwlist

                    is_kw = [i for i in df.columns if i in kwlist]

                    for kwc in is_kw:
                        df[kwc + "__00remove00"] = df[kwc]
                        formula = formula.replace(kwc, kwc + "__00remove00")

                    result = df.eval(formula)
                    for rmcol in df.columns:
                        if "__00remove00" in rmcol:
                            df = df.drop(rmcol, axis=1)

                df[column_name] = result

                df.to_pickle(pickle_path)

                preview_html = (
                    df.sample(n=45)
                    .reset_index(drop=True)
                    .to_html(classes="table table-striped")
                )

                col_type = str(df[column_name].dtype)

                col_stats = {
                    "type": col_type,
                    "missing": float(df[column_name].isna().sum()),
                    "unique_values": float(df[column_name].nunique()),
                }

                if col_type.startswith(("int", "float")):
                    col_values = df[column_name].astype(float)

                    col_stats.update(
                        {
                            "mean": col_values.mean(),
                            "median": col_values.median(),
                            "std": col_values.std(),
                            "min": col_values.min(),
                            "max": col_values.max(),
                        }
                    )

                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                stats = get_stats(df)

                return {
                    "type": "column_added",
                    "preview": preview_html,
                    "column_name": column_name,
                    "column_stats": col_stats,
                    "stats": stats,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "success": True,
                }

            except Exception as formula_error:
                return {
                    "type": "error",
                    "message": f"Formula error: {str(formula_error)}",
                    "success": False,
                }

        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def modify_column(self, file_id, column_name, formula):
        try:
            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            if column_name not in df.columns:
                return {
                    "type": "error",
                    "message": f"Column '{column_name}' doesn't exist.",
                    "success": False,
                }

            original_column = df[column_name].copy()

            try:
                if formula.startswith("="):
                    formula = formula[1:]
                    for col in df.columns:
                        if col != column_name:
                            formula = formula.replace(col, f"df['{col}']")
                    result = eval(formula)
                else:
                    result = df.eval(formula)

                df[column_name] = result

                df.to_pickle(pickle_path)

                preview_html = (
                    df.sample(n=45)
                    .reset_index(drop=True)
                    .to_html(classes="table table-striped")
                )

                col_type = str(df[column_name].dtype)
                col_stats = {
                    "type": col_type,
                    "missing": df[column_name].isna().sum(),
                    "unique_values": df[column_name].nunique(),
                }

                if col_type.startswith(("int", "float")):
                    col_stats.update(
                        {
                            "mean": df[column_name].mean(),
                            "median": df[column_name].median(),
                            "std": df[column_name].std(),
                            "min": df[column_name].min(),
                            "max": df[column_name].max(),
                        }
                    )

                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

                return {
                    "type": "column_modified",
                    "preview": preview_html,
                    "column_name": column_name,
                    "column_stats": col_stats,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "success": True,
                }

            except Exception as formula_error:
                df[column_name] = original_column
                df.to_pickle(pickle_path)

                return {
                    "type": "error",
                    "message": f"Formula error: {str(formula_error)}",
                    "success": False,
                }

        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}

    @database_sync_to_async
    def delete_column(self, file_id, column_name):
        try:
            pickle_path = os.path.join(settings.MEDIA_ROOT, f"temp_{file_id}.pkl")
            df = pd.read_pickle(pickle_path)

            if column_name not in df.columns:
                return {
                    "type": "error",
                    "message": f"Column '{column_name}' doesn't exist.",
                    "success": False,
                }

            df = df.drop(columns=[column_name])

            df.to_pickle(pickle_path)

            preview_html = (
                df.sample(n=45)
                .reset_index(drop=True)
                .to_html(classes="table table-striped")
            )

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            stats = get_stats(df)

            return {
                "type": "column_deleted",
                "preview": preview_html,
                "column_name": column_name,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "stats": stats,
                "success": True,
            }

        except Exception as e:
            return {"type": "error", "message": str(e), "success": False}
