from static_strings_for_data_extraction import *
import requests
import zipfile
import io
import pandas as pd
import numpy as np


def clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces empty strings with NaN then converts all columns of
    dtype 'object' to Int64, float64 or nullable boolean if possible.

    Returns the cleaned DataFrame.
    """
    df = df.replace(r'^\s*$', np.nan, regex = True)

    for col in df.select_dtypes(include = ['object']).columns:

        non_null = df[col].dropna()

        # If they non-null values are all True/False → boolean nullable
        if non_null.isin([True, False]).all():
            df[col] = df[col].astype('boolean')
            continue

        # Otherwise try digital conversion
        numeric = pd.to_numeric(df[col], errors = 'coerce')
        if numeric.notna().any():
            df[col] = numeric
            non_null = df[col].dropna()
            # If all non-nulls are integers, we switch to Int64
            if (non_null.apply(float.is_integer)).all():
                df[col] = df[col].astype('Int64')

        # Otherwise keep object (free text)
    return df

def merge_all_dataFrames(daily_consumption_df ,daily_national_temperature_df ,bank_holidays_df ,school_holidays_df) -> pd.DataFrame:
    """
    Merge consumption, temperature and holiday data into a single DataFrame,
    ensuring that all dates from all sources are included.

    Returns the final DataFrame with:
        - every date present in any input DataFrame,
        - merged consumption and temperature columns,
        - boolean 'Bank holidays' flag,
        - boolean 'School holidays' flag,
        sorted by date and reindexed from 0 to N-1.
    """
    for df, col in [(daily_consumption_df, 'Date'),(daily_national_temperature_df, 'Date'),(bank_holidays_df, 'date'),
        (school_holidays_df, 'date')
    ]:
        df[col] = pd.to_datetime(df[col], format = "mixed").dt.date

    # how = 'outer' guarantees to preserve any date present in either DataFrame
    consump_temp_df = (
        daily_consumption_df
        .merge(
            daily_national_temperature_df[['Date','Avg_temp_min','Avg_temp_max','Avg_temp_mean']],
            on = 'Date', how = 'outer'
        )
    )

    bank_df = (
        bank_holidays_df[['date']]
        .drop_duplicates()
        .assign(**{'Bank holidays': True})
        .rename(columns={'date':'Date'})
    )

    holidays_df = (
        school_holidays_df
        .loc[
            school_holidays_df[['vacances_zone_a','vacances_zone_b','vacances_zone_c']].any(axis = 1),
            ['date']]
        .drop_duplicates()
        .assign(**{'School holidays': True})
        .rename(columns = {'date':'Date'})
    )

    # how = 'outer' guarantees to preserve any date present in either DataFrame
    final_df = (
        consump_temp_df
        .merge(bank_df, on = 'Date', how = 'outer')
        .merge(holidays_df, on = 'Date', how = 'outer')
    )

    # Dates that were not vacations have NaN. They will be changed to False
    final_df['Bank holidays'] = final_df['Bank holidays'].fillna(False)
    final_df['School holidays'] = final_df['School holidays'].fillna(False)

    final_df = clean_object_columns(final_df)

    # Reset the index because when merging the original index is retained
    final_df = final_df.sort_values('Date').reset_index(drop = True)
    return final_df


def compute_national_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the national‐level daily averages of min, max and mean temperature.

    Returns the daily national temperature DataFrames.
    """
    # Delete useless col for national analysis
    df2 = df.drop(columns = ["__id", "Code INSEE département", "Département"])
    df2.columns = ["date", "temp_min", "temp_max", "temp_mean"]

    df2["date"] = pd.to_datetime(df2["date"], errors = "coerce").dt.date

    for col in ["temp_min","temp_max","temp_mean"]:
        df2[col] = pd.to_numeric(df2[col], errors = "coerce")

    # Group by date and compute the mean of each temperature field
    daily_national_temperature_df = (
        df2
        .groupby('date', as_index = False)
        .agg({
            "temp_min":  'mean',
            "temp_max":  'mean',
            "temp_mean": 'mean',
        })
    )

    # Rename columns for clarity
    daily_national_temperature_df.columns = [
        'Date',
        'Avg_temp_min',
        'Avg_temp_max',
        'Avg_temp_mean',
    ]

    return daily_national_temperature_df

def download_and_extract_data_from_csv(url: str) -> pd.DataFrame:
    """
    Download and extract the CSV file in a DataFrame.

    Returns a list of pandas DataFrames (one in this case).
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
        buffer.seek(0)
        df = pd.read_csv(buffer, sep = ',', header = 0, index_col = False)
        print(f"Processed {url}, 1 sheet(s) read.")
        return df
    except Exception as e:
        print(f"Error processing {url}: {e}")

def handle_data_gouv_files() -> pd.DataFrame:
    """
    Download and prepare DataFrames for daily national temperatures and French bank holidays.

    Returns the daily national temperature_df and the bank holidays_df DataFrames.
    """
    daily_departmental_temperature_df = download_and_extract_data_from_csv(FRANCE_DAILY_DEPARTMENTAL_TEMPERATURE_URL)
    daily_national_temperature_df = compute_national_temperatures(daily_departmental_temperature_df)

    bank_holidays_df = download_and_extract_data_from_csv(FRANCE_BANK_HOLIDAYS_URL)

    return daily_national_temperature_df, bank_holidays_df


def download_and_extract_data_from_rte_france(url : str) -> list[pd.DataFrame]:
    """
    Download a ZIP file into memory and extract the corresponding CSV file in a DataFrame.

    Returns a list of pandas DataFrames (in case where there are several files in the archive).
    """
    response = requests.get(url, allow_redirects = True)
    response.raise_for_status()

    zip_bytes = io.BytesIO(response.content)
    z = zipfile.ZipFile(zip_bytes)

    dfs = []
    for fname in z.namelist():
        raw = z.read(fname)
        text = raw.decode(encoding = "latin-1", errors = "replace")
        df = pd.read_csv(io.StringIO(text), sep = '\t', header = 0, index_col = False, low_memory = False)

        # Inspect the first cell of the last line to delete legal prefixe
        if not df.empty:
            first_cell = df.iloc[-1, 0]
            if isinstance(first_cell, str) and any(first_cell.startswith(p) for p in LEGAL_PREFIX):
                df = df.iloc[:-1].reset_index(drop = True)

        dfs.append(df)

    return dfs

def concatenate_dataframe(urls : list[str]) -> pd.DataFrame:
    """
    Download multiple ZIP files into memory, extract the corresponding CSV files in DataFrames,
    and combine them into a single DataFrame.

    Returns a single DataFrame obtained by concatenating all sheets from all URLs.
    If no sheets were successfully read, returns an empty DataFrame.
    """
    dfs = []

    for url in urls:
        try:
            df = download_and_extract_data_from_rte_france(url)
            dfs.extend(df)
            print(f"Processed {url}, {len(df)} sheet(s) read.")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return pd.concat(dfs, ignore_index = True) if dfs else pd.DataFrame()

def handle_RTE_france_files() -> pd.DataFrame:
    """
    Download, extract and merge RTE France consumption and TEMPO calendars into one DataFrame.

    Returns a DataFrame containing consumption calendar data augmented with the 'Type de jour TEMPO' column from
    the TEMPO calendar.
    """
    # Download, extract, read and merge all consumption calendar files
    daily_consumption_df = concatenate_dataframe(FRANCE_CONSUMPTION_CALENDAR_STATIC_URLS + FRANCE_CONSUMPTION_CALENDAR_URLS)

    # Download, extract, read and merge all tempo calendar files
    daily_tempo_df = concatenate_dataframe(FRANCE_TEMPO_CALENDAR_URLS)

    # Convert 'Date' in datetime.date to ensure a reliable matching
    daily_consumption_df['Date'] = pd.to_datetime(daily_consumption_df['Date'], format = "mixed").dt.date
    daily_tempo_df['Date'] = pd.to_datetime(daily_tempo_df['Date'], format = "mixed").dt.date

    # Merge the tempo col of the tempo df in the consumption df
    daily_consumption_with_tempo_df = daily_consumption_df.merge(
        daily_tempo_df[['Date', 'Type de jour TEMPO']],
        on = 'Date',
        how = 'left'
    )

    return daily_consumption_with_tempo_df


if __name__=='__main__':

    daily_consumption_df = handle_RTE_france_files()

    daily_national_temperature_df, bank_holidays_df = handle_data_gouv_files()

    school_holidays_df = download_and_extract_data_from_csv(FRANCE_SCHOOL_HOLIDAYS_URL)

    final_df = merge_all_dataFrames(daily_consumption_df, daily_national_temperature_df, bank_holidays_df, school_holidays_df)

    output_path = "./final_df.csv"
    final_df.to_csv("./final_df.csv", index = False)
    print(f"DataFrame enregistré sous : {output_path}")