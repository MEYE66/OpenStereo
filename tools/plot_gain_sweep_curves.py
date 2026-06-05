import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot gain-sweep performance curves from multiple sweep csv files.'
    )
    parser.add_argument(
        '--input-csvs',
        type=str,
        nargs='+',
        required=True,
        help='One or more gain_sweep_metrics.csv files to plot together.')
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Target image path, for example output/plots/gain_sweep.png.')
    parser.add_argument(
        '--title',
        type=str,
        default='Gain Sweep vs Stereo Matching Performance',
        help='Figure title.')
    parser.add_argument(
        '--label-field',
        type=str,
        default='gaussian_var',
        choices=['gaussian_var', 'exposure_ms'],
        help='Row field used to label each plotted curve.')
    return parser.parse_args()


def format_series_label(label_field, label_value):
    if label_field == 'exposure_ms':
        return f'exposure_ms={label_value:g}'
    return f'gaussian_var={label_value:g}'


def read_sweep_csv(path, label_field):
    with open(path, 'r', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No rows found in {path}')

    label_value = float(rows[0][label_field])
    points = []
    for row in rows:
        points.append({
            'gain': float(row['gain']),
            'd1_all': float(row['d1_all']),
            'epe': float(row['epe']),
        })
    points.sort(key=lambda item: item['gain'])
    return label_value, points


def try_read_reference(csv_path):
    reference_path = Path(csv_path).resolve().parent / 'reference_metrics.csv'
    if not reference_path.exists():
        return None
    with open(reference_path, 'r', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[0]
    return {
        'd1_all': float(row['d1_all']),
        'epe': float(row['epe']),
    }


def plot_curves(series_list, output_path, title, label_field, reference_metrics=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    metric_specs = [
        ('d1_all', 'D1-all (%)'),
        ('epe', 'EPE'),
    ]

    for axis, (metric_name, metric_label) in zip(axes, metric_specs):
        for label_value, points in series_list:
            gains = [item['gain'] for item in points]
            values = [item[metric_name] for item in points]
            axis.plot(
                gains,
                values,
                marker='o',
                linewidth=2.0,
                markersize=4.5,
                label=format_series_label(label_field, label_value),
            )

        if reference_metrics is not None:
            axis.axhline(
                y=reference_metrics[metric_name],
                color='0.35',
                linestyle='--',
                linewidth=1.5,
                label=f'reference {metric_name}={reference_metrics[metric_name]:.4f}',
            )

        axis.set_xlabel('Gain')
        axis.set_ylabel(metric_label)
        axis.grid(True, linestyle='--', alpha=0.35)
        axis.set_xticks(range(1, 21))
        axis.legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    csv_paths = [Path(path).resolve() for path in args.input_csvs]

    series_list = []
    for csv_path in csv_paths:
        label_value, points = read_sweep_csv(csv_path, args.label_field)
        series_list.append((label_value, points))
    series_list.sort(key=lambda item: item[0])

    reference_metrics = try_read_reference(csv_paths[0])
    output_path = Path(args.output_path).resolve()
    plot_curves(
        series_list,
        output_path,
        args.title,
        args.label_field,
        reference_metrics=reference_metrics,
    )

    print(f'[plot] saved figure to {output_path}', flush=True)


if __name__ == '__main__':
    main()