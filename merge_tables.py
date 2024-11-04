import argparse
import pandas as pd


def merge_tables(pickles: list[str], output: str):
    df = pd.concat([pd.read_pickle(p) for p in pickles])
    df.sort_index(axis=1, inplace=True)
    df.fillna('N/A', inplace=True)

    col_fmt = 'l' * df.index.nlevels
    metric_type = None
    for col in df.columns:
        if col[0] != metric_type:
            col_fmt += '|'
            metric_type = col[0]
        col_fmt += 'c'
    df.style.to_latex(
        buf=output,
        column_format=col_fmt,
        multicol_align='c|',
        hrules=True,
        clines="skip-last;data",
        multirow_align='t'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--tables', type=str, nargs='+', required=True)
    args = parser.parse_args()
    merge_tables(args.tables, args.output)

