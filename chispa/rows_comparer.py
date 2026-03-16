from __future__ import annotations

from collections.abc import Callable
from itertools import zip_longest
from typing import Any

from prettytable import PrettyTable
from pyspark.sql import Row

import chispa
from chispa.common_enums import DataFrameDiffOutputFormat
from chispa.formatting import FormattingConfig, format_string
from chispa.pretty_diff import cell_to_string, show_diff_separate_lines, show_diff_side_by_side


def assert_basic_rows_equality(
    rows1: list[Row], rows2: list[Row], underline_cells: bool = False,
    formats: FormattingConfig | None = None, columns: list[str] | None = None,
) -> None:
    if not formats:
        formats = FormattingConfig()
    elif not isinstance(formats, FormattingConfig):
        formats = FormattingConfig._from_arbitrary_dataclass(formats)

    if rows1 != rows2:
        if formats.output_format == DataFrameDiffOutputFormat.SEPARATE_LINES and columns is not None:
            all_rows_equal, msg = show_diff_separate_lines(columns, rows1, rows2, formats=formats)
            if not all_rows_equal:
                raise chispa.DataFramesNotEqualError("\n" + msg)
            return

        all_rows_equal, msg = show_diff_side_by_side(rows1, rows2, formats=formats)
        if not all_rows_equal:
            raise chispa.DataFramesNotEqualError("\n" + msg)


def assert_generic_rows_equality(
    rows1: list[Row],
    rows2: list[Row],
    row_equality_fun: Callable,  # type: ignore[type-arg]
    row_equality_fun_args: dict[str, Any],
    underline_cells: bool = False,
    formats: FormattingConfig | None = None,
    columns: list[str] | None = None,
) -> None:
    if not formats:
        formats = FormattingConfig()
    elif not isinstance(formats, FormattingConfig):
        formats = FormattingConfig._from_arbitrary_dataclass(formats)

    df1_rows = rows1
    df2_rows = rows2

    if formats.output_format == DataFrameDiffOutputFormat.SEPARATE_LINES and columns is not None:
        any_mismatch = any(
            (r1 is None) != (r2 is None) or (r1 is not None and not row_equality_fun(r1, r2, **row_equality_fun_args))
            for r1, r2 in zip_longest(df1_rows, df2_rows)
        )
        if any_mismatch:
            _, msg = show_diff_separate_lines(columns, rows1, rows2, formats=formats)
            raise chispa.DataFramesNotEqualError("\n" + msg)
        return

    zipped = list(zip_longest(df1_rows, df2_rows))
    t = PrettyTable(["df1", "df2"])
    row_results = [
        _generic_row_result(r1, r2, row_equality_fun, row_equality_fun_args, formats)
        for r1, r2 in zipped
    ]
    for table_row, _ in row_results:
        t.add_row(table_row)
    all_rows_equal = all(eq for _, eq in row_results)
    if not all_rows_equal:
        raise chispa.DataFramesNotEqualError("\n" + t.get_string())


def _generic_row_result(
    r1: object | None, r2: object | None,
    row_equality_fun: Callable, row_equality_fun_args: dict[str, Any],  # type: ignore[type-arg]
    formats: FormattingConfig,
) -> tuple[list, bool]:
    """Returns (table_row, is_equal) for a pair of rows."""
    if (r1 is None) ^ (r2 is None):
        return [
            format_string(str(r1), formats.mismatched_rows),
            format_string(str(r2), formats.mismatched_rows),
        ], False

    if row_equality_fun(r1, r2, **row_equality_fun_args):
        r1_string = ", ".join(f"{f}={cell_to_string(r1[f])}" for f in r1.__fields__)
        r2_string = ", ".join(f"{f}={cell_to_string(r2[f])}" for f in r2.__fields__)
        return [
            format_string(r1_string, formats.matched_rows),
            format_string(r2_string, formats.matched_rows),
        ], True

    r_zipped = list(zip_longest(r1.__fields__, r2.__fields__))
    mismatches = [r1[f1] != r2[f2] for f1, f2 in r_zipped]
    r1_string_list = [
        format_string(f"{f1}={cell_to_string(r1[f1])}", formats.mismatched_cells if m else formats.matched_cells)
        for (f1, _), m in zip(r_zipped, mismatches)
    ]
    r2_string_list = [
        format_string(f"{f2}={cell_to_string(r2[f2])}", formats.mismatched_cells if m else formats.matched_cells)
        for (_, f2), m in zip(r_zipped, mismatches)
    ]
    return [", ".join(r1_string_list), ", ".join(r2_string_list)], not any(mismatches)
