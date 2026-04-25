import pandas as pd

def get_rich_monthly_nodes(city, start, end, target_month=0):

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    # Load the datasets
    tg_df = pd.read_csv(f'../../data/tg/{city}_tg.csv')
    tn_df = pd.read_csv(f'../../data/tn/{city}_tn.csv')
    tx_df = pd.read_csv(f'../../data/tx/{city}_tx.csv')
    rr_df = pd.read_csv(f'../../data/rr/{city}_rr.csv')
    qq_df = pd.read_csv(f'../../data/qq/{city}_qq.csv')
    hu_df = pd.read_csv(f'../../data/hu/{city}_hu.csv')


    # Convert 'time' column to datetime objects
    tg_df['time'] = pd.to_datetime(tg_df['time'])
    tn_df['time'] = pd.to_datetime(tn_df['time'])
    tx_df['time'] = pd.to_datetime(tx_df['time'])
    rr_df['time'] = pd.to_datetime(rr_df['time'])
    qq_df['time'] = pd.to_datetime(qq_df['time'])
    hu_df['time'] = pd.to_datetime(hu_df['time'])

    # Set 'time' as the index
    tg_df.set_index('time', inplace=True)
    tn_df.set_index('time', inplace=True)
    tx_df.set_index('time', inplace=True)
    rr_df.set_index('time', inplace=True)
    qq_df.set_index('time', inplace=True)
    hu_df.set_index('time', inplace=True)

    # Merge the dataframes
    df = tg_df.join(tn_df).join(tx_df).join(rr_df).join(qq_df).join(hu_df)

    # Resample the data by month and aggregate
    monthly_nodes = {}
    for name, group in df.groupby(pd.Grouper(freq='ME')):

        if target_month != 0 and name.month != target_month:
            continue

        if name < start_date or name.to_period('M') > end_date.to_period('M'):
            continue
        

        month_str = name.strftime('%Y-%m')
        
        # Calculate the mean of tn and tx for the month
        mean_tn = group['tn'].mean()
        mean_tx = group['tx'].mean()
        mean_tg = group['tg'].mean()
        rr_sum = group['rr'].sum()
        mean_qq = group['qq'].mean()
        mean_hu = group['hu'].mean()


        metrics = [mean_tn, mean_tx, mean_tg, rr_sum, mean_qq, mean_hu]
        if any(pd.isna(m) for m in metrics):
            # print(f"Skipping month {month_str} due to missing data.")
            continue
        
        monthly_nodes[month_str] = {
            'mean_tn': mean_tn,
            'mean_tx': mean_tx,
            'mean_tg': mean_tg,
            'rr_sum': rr_sum,
            'mean_qq': mean_qq,
            'mean_hu': mean_hu
        }

    return monthly_nodes
