import pandas as pd
import glob
import os

def load_and_prepare_data(raw_path, weeks_to_load=None):

    train_folder = os.path.join(raw_path, 'train', 'weeks')
    all_week_files = sorted(glob.glob(os.path.join(train_folder, "*.parquet")))

    if not all_week_files:
        print(f"ОШИБКА: Файлы не найдены в {train_folder}")
        return None

    if weeks_to_load:
        all_week_files = [f for f in all_week_files if any(w in f for w in weeks_to_load)]

    print(f"Загружаю {len(all_week_files)} файлов из {train_folder}...")

    df = pd.concat((pd.read_parquet(f, engine='fastparquet') for f in all_week_files), ignore_index=True)

    meta_path = os.path.join(raw_path, 'items_metadata.parquet')
    if os.path.exists(meta_path):
        item_meta = pd.read_parquet(meta_path, engine='fastparquet')
        df = pd.merge(df, item_meta, on='item_id', how='left')
        print("Метаданные успешно приклеены.")
    else:
        print("ПРЕДУПРЕЖДЕНИЕ: item_meta.parquet не найден.")

    users_meta_path = os.path.join(raw_path, 'users_metadata.parquet')
    if os.path.exists(users_meta_path):
        users_meta = pd.read_parquet(users_meta_path, engine='fastparquet')
        # Склеиваем по user_id
        df = pd.merge(df, users_meta, on='user_id', how='left')
        print("Данные о пользователях (пол, возраст, гео) добавлены.")
    else:
        print("Внимание: users_metadata.parquet не найден!")

    return df

def calculate_target(df):

    col_duration = 'duration' if 'duration' in df.columns else 'video_duration'

    if col_duration in df.columns and 'timespent' in df.columns:
        df['target'] = (df['timespent'].astype(float) / df[col_duration].astype(float)).fillna(0)
        df.loc[df['target'] > 1, 'target'] = 1.0

        print(f"Целевая переменная 'target' рассчитана успешно через колонку '{col_duration}'.")
    else:
        available = list(df.columns)
        print(f"ОШИБКА: Не найдены нужные столбцы! В наличии только: {available}")

    return df

def save_processed_data(df, output_path, filename):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_path = os.path.join(output_path, filename)
    df.to_parquet(full_path, index=False, engine='fastparquet')
    print(f"Файл сохранен: {full_path}")