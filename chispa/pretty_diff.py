"""
Utility for rendering DataFrame row diffs in SeparateLines format.

Ported from spark-fast-tests ProductUtil.showProductDiffSeparateLine.
"""
from __future__ import annotations

from itertools import zip_longest

from prettytable import PrettyTable

from chispa.formatting import FormattingConfig, format_string


def cell_to_string(cell: object, truncate: int = 500) -> str:
    """Convert a cell value to a display string, handling nulls, bytes, nested Rows, etc."""
    if cell is None:
        return "null"
    if isinstance(cell, (bytes, bytearray)):
        return "[" + " ".join(f"{b:02X}" for b in cell) + "]"
    # Check Row before tuple since PySpark Row extends tuple
    try:
        from pyspark.sql.types import Row

        if isinstance(cell, Row) and hasattr(cell, "__fields__"):
            return "{" + ", ".join(f"{k} -> {cell[k]}" for k in cell.__fields__) + "}"
    except ImportError:
        pass
    if isinstance(cell, (list, tuple)):
        return "[" + ", ".join(str(v) for v in cell) + "]"
    s = str(cell)
    if 0 < truncate < len(s):
        if truncate < 4:
            return s[:truncate]
        return s[: truncate - 3] + "..."
    return s


def show_diff_side_by_side(
    rows1: list, rows2: list, formats: FormattingConfig
) -> tuple[bool, str]:
    """Side-by-side format using PrettyTable."""
    t = PrettyTable(["df1", "df2"])
    zipped = list(zip_longest(rows1, rows2))
    row_results = [_side_by_side_row(r1, r2, formats) for r1, r2 in zipped]
    for row_data in row_results:
        t.add_row(row_data[0])
    all_rows_equal = all(entry[1] for entry in row_results)
    return all_rows_equal, t.get_string()


def _side_by_side_row(
    r1: object | None, r2: object | None, formats: FormattingConfig,
) -> tuple[list, bool]:
    """Returns (table_row, is_equal) for a pair of rows."""
    if r1 is None and r2 is not None:
        return [None, format_string(str(r2), formats.mismatched_rows)], False
    if r1 is not None and r2 is None:
        return [format_string(str(r1), formats.mismatched_rows), None], False
    if r1 is None and r2 is None:
        return [format_string("null", formats.matched_rows), format_string("null", formats.matched_rows)], True

    r_zipped = list(zip_longest(r1.__fields__, r2.__fields__))
    mismatches = [
        (r1[f1] if f1 else None) != (r2[f2] if f2 else None)
        for f1, f2 in r_zipped
    ]
    r1_string = [
        format_string(
            f"{f1}={cell_to_string(r1[f1] if f1 else None)}",
            formats.mismatched_cells if m else formats.matched_cells,
        )
        for (f1, _), m in zip(r_zipped, mismatches)
    ]
    r2_string = [
        format_string(
            f"{f2}={cell_to_string(r2[f2] if f2 else None)}",
            formats.mismatched_cells if m else formats.matched_cells,
        )
        for (_, f2), m in zip(r_zipped, mismatches)
    ]
    return [", ".join(r1_string), ", ".join(r2_string)], not any(mismatches)


def show_diff_separate_lines(
    columns: list[str],
    rows1: list,
    rows2: list,
    truncate: int = 500,
    formats: FormattingConfig | None = None,
) -> tuple[bool, str]:
    """
    SeparateLines format: shows each row as a table row with column values aligned.
    Actual and expected on separate lines with row index, mismatched rows highlighted.
    """
    if not formats:
        formats = FormattingConfig()

    actual_rows = [_row_to_seq(r) for r in rows1]
    expected_rows = [_row_to_seq(r) for r in rows2]
    all_data = actual_rows + expected_rows
    col_widths = _get_col_widths(columns, all_data, truncate, min_col_width=3)

    max_index = max(len(actual_rows), len(expected_rows)) - 1 if (actual_rows or expected_rows) else 0
    index_width = len(f"{max_index}:") + 1 if max_index >= 0 else 2

    sep = " " * index_width + "+".join("-" * w for w in col_widths) + "+\n"
    header_cells = [cell_to_string(c, truncate).rjust(w) for c, w in zip(columns, col_widths)]
    header_line = " " * index_width + "|".join(header_cells) + "|\n"

    zipped = list(zip_longest(actual_rows, expected_rows, fillvalue=[]))
    diff_data = [_build_row_diff(i, actual, expected, col_widths, truncate, formats) for i, (actual, expected) in enumerate(zipped)]
    all_rows_equal = all(entry[2] for entry in diff_data)

    sb = [sep, header_line] + [
        line
        for idx, (actual_line, expected_line, rows_eq, i) in enumerate(diff_data)
        for line in _render_diff_entry(actual_line, expected_line, rows_eq, i, idx, diff_data, index_width, sep)
    ] + [sep]
    return all_rows_equal, "".join(sb)


def _build_row_diff(
    i: int, actual: list, expected: list, col_widths: list[int], truncate: int, formats: FormattingConfig,
) -> tuple[str, str | None, bool, int]:
    if actual == expected:
        actual_cells = _pad_cells(actual, col_widths, truncate)
        actual_str = "|".join(format_string(c, formats.matched_cells) for c in actual_cells)
        return (actual_str, None, True, i)

    if actual and expected:
        pairs = [
            (
                cell_to_string(a, truncate).rjust(col_widths[j]) if j < len(col_widths) else str(a),
                cell_to_string(e, truncate).rjust(col_widths[j]) if j < len(col_widths) else str(e),
                a == e,
            )
            for j, (a, e) in enumerate(zip_longest(actual, expected, fillvalue=None))
        ]
        actual_parts = [format_string(ac, formats.matched_cells if eq else formats.mismatched_cells) for ac, _, eq in pairs]
        expected_parts = [format_string(ec, formats.matched_cells if eq else formats.mismatched_cells) for _, ec, eq in pairs]
        return ("|".join(actual_parts), "|".join(expected_parts), False, i)

    if not actual:
        expected_str = "|".join(
            format_string(cell_to_string(v, truncate).rjust(col_widths[j]), formats.mismatched_cells)
            for j, v in enumerate(expected)
        )
        return ("", expected_str, False, i)

    actual_str = "|".join(
        format_string(cell_to_string(v, truncate).rjust(col_widths[j]), formats.mismatched_cells)
        for j, v in enumerate(actual)
    )
    return (actual_str, "", False, i)


def _render_diff_entry(
    actual_line: str, expected_line: str | None, rows_eq: bool, i: int,
    idx: int, diff_data: list, index_width: int, sep: str,
) -> list[str]:
    index_str = f"{i + 1}:|".rjust(index_width)
    is_last = idx >= len(diff_data) - 1
    return (
        ([f"{index_str}{actual_line}|:{i + 1}\n"] if actual_line else [])
        + ([f"{index_str}{expected_line}|:{i + 1}\n"] if not rows_eq and expected_line else [])
        + ([sep] if not rows_eq and not is_last else [])
        + ([sep] if rows_eq and not is_last and not diff_data[idx + 1][2] else [])
    )


def _row_to_seq(row: object) -> list:
    if row is None:
        return []
    return list(row)  # type: ignore[call-overload]


def _get_col_widths(columns: list[str], rows: list[list], truncate: int, min_col_width: int = 3) -> list[int]:
    header_widths = [max(min_col_width, len(cell_to_string(c, truncate))) for c in columns]
    return [
        max(hw, *(len(cell_to_string(row[i], truncate)) for row in rows if i < len(row)), hw)
        for i, hw in enumerate(header_widths)
    ]


def _pad_cells(row: list, col_widths: list[int], truncate: int) -> list[str]:
    if not row:
        return []
    return [cell_to_string(val, truncate).rjust(col_widths[i]) for i, val in enumerate(row)]
