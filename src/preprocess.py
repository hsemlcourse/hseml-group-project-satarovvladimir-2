import pandas as pd

def clean_data(df):
    initial_rows = len(df)
    df = df[df['duration'] > 0].copy()
    df = df.dropna(subset=['user_id', 'item_id'])
    rows_removed = initial_rows - len(df)
    print(f"Очистка завершена. Удалено строк: {rows_removed} ({(rows_removed / initial_rows) * 100:.2f}%)")
    return df