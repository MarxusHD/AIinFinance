import pandas as pd
import numpy as np

def event_lift_table(df_in: pd.DataFrame, events: list[str], y_col: str) -> pd.DataFrame:
    base = df_in[y_col].astype(float).mean()
    rows=[]
    for e in events:
        if e not in df_in.columns:
            continue
        s = pd.to_numeric(df_in[e], errors="coerce").fillna(0).astype(int)
        if s.sum() == 0:
            continue
        rate = df_in.loc[s==1, y_col].astype(float).mean()
        prev = s.mean()
        rows.append({
            "event": e,
            "prevalence": prev,
            "cond_distress_rate": rate,
            "lift_vs_base": rate/base if base>0 else np.nan,
            "base_rate": base,
            "n_event": int(s.sum()),
        })
    out = pd.DataFrame(rows).sort_values("lift_vs_base", ascending=False)
    return out

# Case 1: Empty events list
df = pd.DataFrame({"target": [0, 1, 0, 1]})
try:
    print("Testing Case 1: Empty events list")
    print(event_lift_table(df, [], "target"))
except Exception as e:
    print(f"Caught expected error: {type(e).__name__}: {e}")

# Case 2: Events not in columns
try:
    print("\nTesting Case 2: Events not in columns")
    print(event_lift_table(df, ["event1"], "target"))
except Exception as e:
    print(f"Caught expected error: {type(e).__name__}: {e}")

# Case 3: Event occurrences are zero
df["event1"] = [0, 0, 0, 0]
try:
    print("\nTesting Case 3: Event occurrences are zero")
    print(event_lift_table(df, ["event1"], "target"))
except Exception as e:
    print(f"Caught expected error: {type(e).__name__}: {e}")
