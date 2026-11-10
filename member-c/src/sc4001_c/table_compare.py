"""Strict table comparison with a narrow allowance for SciPy roundoff.

Input evidence hashes, predictions, categorical fields and exact randomization
tests remain exact. Only approximate t probabilities/intervals may differ by
at most 1e-12 in absolute units; every such cell is recorded for inspection.
"""
import csv
import math

APPROXIMATE_COLUMNS = {'ci95_low', 'ci95_high', 'p_t_approx', 'p_t_approx_holm'}
ABSOLUTE_TOLERANCE = 1e-12


def compare_tables(reference, observed):
    with reference.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        expected = list(reader)
    with observed.open(newline='') as stream:
        reader = csv.DictReader(stream)
        actual_columns = reader.fieldnames
        actual = list(reader)
    if columns != actual_columns or len(expected) != len(actual):
        return {'equivalent': False, 'reason': 'Changed columns or row count'}
    differences = []
    for row_number, (left, right) in enumerate(zip(expected, actual), start=2):
        if (set(left) != set(columns) or set(right) != set(columns)
                or any(value is None for value in left.values())
                or any(value is None for value in right.values())):
            return {'equivalent': False, 'reason': 'Malformed CSV row', 'csv_line': row_number}
        for column in columns:
            if left[column] == right[column]:
                continue
            record = {'csv_line': row_number, 'column': column,
                      'reference': left[column], 'observed': right[column]}
            if column not in APPROXIMATE_COLUMNS:
                return {'equivalent': False, 'reason': 'Changed exact field', 'difference': record}
            try:
                a, b = float(left[column]), float(right[column])
            except (ValueError, TypeError):
                return {'equivalent': False, 'reason': 'Invalid approximate value', 'difference': record}
            record['absolute_difference'] = abs(a - b)
            if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > ABSOLUTE_TOLERANCE:
                return {'equivalent': False, 'reason': 'Approximate value differs beyond roundoff', 'difference': record}
            differences.append(record)
    return {'equivalent': True, 'rows': len(actual), 'roundoff_differences': differences,
            'max_absolute_difference': max((r['absolute_difference'] for r in differences), default=0.0),
            'absolute_tolerance': ABSOLUTE_TOLERANCE,
            'approximate_columns': sorted(APPROXIMATE_COLUMNS)}
