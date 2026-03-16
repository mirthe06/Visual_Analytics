import pandas as pd
import os
import glob

def process_year_folder(folder_path, year_prefix):
    """
    Processes all CSV files in a folder for a specific year, aggregates by datetime, 
    and merges them into a single dataframe.
    """
    pollutant_dfs = []
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None

    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            # Extract pollutant name (e.g., '2023_BC.csv' -> 'BC')
            # Removing the year prefix and extension
            pollutant = filename.replace(f"{year_prefix}_", "").replace(".csv", "")
            
            print(f"Processing {year_prefix} - {pollutant}...")
            
            # Read the CSV file
            # Using comment='#' skips all metadata lines starting with #
            df = pd.read_csv(
                file_path,
                sep=";",
                comment="#",
                engine="python",
                on_bad_lines="skip"
            )
            
            # Clean column names (strip whitespace and lower case for easier matching)
            df.columns = df.columns.str.strip().str.lower()
            
            # Find the specific columns we need
            # 'begindatumtijd' and 'waarde'
            time_cols = [c for c in df.columns if "begindatumtijd" in c]
            value_cols = [c for c in df.columns if "waarde" in c]
            
            if not time_cols or not value_cols:
                print(f"  Warning: Target columns ('begindatumtijd', 'waarde') not found in {filename}. Skipping.")
                continue
            
            time_col = time_cols[0]
            value_col = value_cols[0]
            
            # Keep only the necessary columns
            df = df[[time_col, value_col]]
            
            # Rename columns to standard names
            df = df.rename(columns={
                time_col: "datetime",
                value_col: pollutant
            })
            
            # Convert datetime column to proper timestamp
            # The format is ISO8601 with timezone, e.g., 2023-01-01T00:00:00+01:00
            # We convert to 'Europe/Amsterdam' time and then strip the timezone 
            # to match the user's expected format (naive datetime).
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            
            # Ensure it's in the correct timezone then make it naive for the example output format
            try:
                # Some might be already naive or have different offsets, convert consistently
                df["datetime"] = df["datetime"].dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
            except TypeError:
                # If they were already naive, just ensure they are naive
                df["datetime"] = df["datetime"].dt.tz_localize(None)
            
            # Drop rows with malformed datetime
            df = df.dropna(subset=["datetime"])
            
            # Handle malformed 'waarde' (convert to numeric, errors become NaN)
            df[pollutant] = pd.to_numeric(df[pollutant], errors="coerce")
            
            # Aggregation: Handle multiple stations by taking the mean for each timestamp
            df = df.groupby("datetime", as_index=False)[pollutant].mean()
            
            pollutant_dfs.append(df)
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    if not pollutant_dfs:
        return None

    # Merge all pollutant dataframes for this year on 'datetime'
    merged_year_df = pollutant_dfs[0]
    for next_df in pollutant_dfs[1:]:
        merged_year_df = pd.merge(merged_year_df, next_df, on="datetime", how="outer")
    
    return merged_year_df

def main():
    base_dir = "/Users/yeswanth/Desktop/VA/Dataset"
    
    # Process 2023
    print("Starting processing for 2023...")
    df_2023 = process_year_folder(os.path.join(base_dir, "2023"), "2023")
    if df_2023 is not None:
        df_2023 = df_2023.sort_values("datetime")
        df_2023.to_csv(os.path.join(base_dir, "cleaned_air_quality_2023.csv"), index=False)
        print("Completed 2023.")
    
    # Process 2024
    print("\nStarting processing for 2024...")
    df_2024 = process_year_folder(os.path.join(base_dir, "2024"), "2024")
    if df_2024 is not None:
        df_2024 = df_2024.sort_values("datetime")
        df_2024.to_csv(os.path.join(base_dir, "cleaned_air_quality_2024.csv"), index=False)
        print("Completed 2024.")
    
    # Combine both years
    if df_2023 is not None and df_2024 is not None:
        print("\nMerging 2023 and 2024 data...")
        final_df = pd.concat([df_2023, df_2024], axis=0, ignore_index=True)
    elif df_2023 is not None:
        final_df = df_2023
    elif df_2024 is not None:
        final_df = df_2024
    else:
        print("No data found to merge.")
        return

    # Sort final dataframe by datetime
    final_df = final_df.sort_values("datetime")
    
    # Drop duplicates if any (e.g. if files overlap)
    final_df = final_df.drop_duplicates(subset=["datetime"])
    
    # Save the final result
    output_path = os.path.join(base_dir, "cleaned_air_quality_2023_2024.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Final cleaned dataset saved to: {output_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"Columns: {list(final_df.columns)}")

if __name__ == "__main__":
    main()
