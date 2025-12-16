import pandas as pd

def get_monthly_nodes(city):
    # Load the datasets
    tg_df = pd.read_csv(f'../../data/tg/{city}_tg.csv')
    tn_df = pd.read_csv(f'../../data/tn/{city}_tn.csv')
    tx_df = pd.read_csv(f'../../data/tx/{city}_tx.csv')

    # Convert 'time' column to datetime objects
    tg_df['time'] = pd.to_datetime(tg_df['time'])
    tn_df['time'] = pd.to_datetime(tn_df['time'])
    tx_df['time'] = pd.to_datetime(tx_df['time'])

    # Set 'time' as the index
    tg_df.set_index('time', inplace=True)
    tn_df.set_index('time', inplace=True)
    tx_df.set_index('time', inplace=True)

    # Merge the dataframes
    df = tg_df.join(tn_df).join(tx_df)

    # Calculate the daily derivative of the average temperature
    df['tg_derivative'] = df['tg'].diff()

    # Resample the data by month and aggregate
    monthly_nodes = {}
    for name, group in df.groupby(pd.Grouper(freq='ME')):  # Corrected from 'M' to 'ME'
        month_str = name.strftime('%Y-%m')
        
        # Get the list of daily derivatives, dropping the first NaN value
        tg_derivatives = [round(x, 1) for x in group['tg_derivative'].dropna().tolist()]
        
        # Calculate the mean of tn and tx for the month
        mean_tn = group['tn'].mean()
        mean_tx = group['tx'].mean()
        mean_tg = group['tg'].mean()
        
        monthly_nodes[month_str] = {
            'tg_derivatives': tg_derivatives,
            'mean_tn': mean_tn,
            'mean_tx': mean_tx,
            'mean_tg': mean_tg
        }

    return monthly_nodes