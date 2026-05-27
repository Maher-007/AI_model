import re
import pandas as pd

def _col_exists(df, col):
    return col in df.columns

def _extract_col(text, df):
    # try to find a column name mentioned in the question by exact match or case-insensitive
    for col in df.columns:
        if re.search(r"\b" + re.escape(col) + r"\b", text, flags=re.IGNORECASE):
            return col
    # fallback: if there's a single word that matches after 'of'
    m = re.search(r"of\s+(\w+)", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1)
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    return None

def answer_query(df: pd.DataFrame, question: str):
    """Very small rule-based NL->pandas responder.

    Supports queries like:
    - "mean of column"
    - "median of column"
    - "min/max of column"
    - "how many rows"
    - "count where column = value"
    - "unique values of column"
    - "distribution of column"
    - "correlation between col1 and col2"
    - "show rows where column > 5"
    - "describe column"
    """
    q = question.strip().lower()

    if not q:
        return "Please enter a question."

    # how many rows
    if re.search(r"how many rows|number of rows|row count", q):
        return f"Rows: {len(df)}"

    # describe
    if re.search(r"describe|summary|statistics", q):
        return df.describe().to_dict()

    # mean
    if "mean" in q or "average" in q:
        col = _extract_col(q, df)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return {"mean": float(df[col].mean())}
        return "Can't find a numeric column to compute mean."

    # median
    if "median" in q:
        col = _extract_col(q, df)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return {"median": float(df[col].median())}
        return "Can't find a numeric column to compute median."

    # min/max
    if "min" in q or "maximum" in q or "maximum" in q or "max" in q:
        col = _extract_col(q, df)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return {"min": float(df[col].min()), "max": float(df[col].max())}
        return "Can't find a numeric column to compute min/max."

    # unique values
    if re.search(r"unique values|unique|distinct", q):
        col = _extract_col(q, df)
        if col:
            return {"unique": df[col].unique().tolist()}
        return "Specify a column to list unique values."

    # count where
    m = re.search(r"count where (\w+)\s*(=|==|>)\s*('?\w+'?)", q)
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3)
        val = val.strip("'\"")
        if col in df.columns:
            try:
                series = df[col].astype(type(df[col].dropna().iloc[0]))
            except Exception:
                series = df[col]
            if op in ("=", "=="):
                return int((series == val).sum())
            if op == ">":
                return int((series.astype(float) > float(val)).sum())
        return "Couldn't compute the count for that condition."

    # correlation
    m = re.search(r"correlation between (\w+) and (\w+)", q)
    if m:
        c1, c2 = m.group(1), m.group(2)
        if c1 in df.columns and c2 in df.columns:
            return {"corr": float(df[c1].corr(df[c2]))}
        return "Columns not found for correlation."

    # show rows where
    m = re.search(r"show rows where (\w+)\s*(=|==|>|<)\s*('?\w+'?)", q)
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3)
        val = val.strip("'\"")
        if col in df.columns:
            series = df[col]
            try:
                if op in ("=", "=="):
                    res = df[series == val]
                elif op == ">":
                    res = df[series.astype(float) > float(val)]
                elif op == "<":
                    res = df[series.astype(float) < float(val)]
                else:
                    return "Operator not supported."
                return res.head().to_dict(orient='records')
            except Exception:
                return "Error filtering rows — check types and column name."

    return "Sorry, I don't understand that question. Try: 'mean of <column>', 'how many rows', 'unique values of <column>', 'show rows where <col> = value', or 'correlation between <col1> and <col2>'."
