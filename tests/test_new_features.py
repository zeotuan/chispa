"""Tests for new features ported from spark-fast-tests v3.0.1+."""
from __future__ import annotations

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from chispa import (
    DataFrameDiffOutputFormat,
    DataFramesNotEqualError,
    FormattingConfig,
    OutputFormat,
    SchemasNotEqualError,
    assert_df_equality,
)
from chispa.pretty_diff import cell_to_string, show_diff_separate_lines, show_diff_side_by_side
from chispa.schema_comparer import assert_schema_equality
from chispa.schema_tree_comparer import (
    FieldComparison,
    FieldInfo,
    are_fields_equal,
    build_comparison,
    tree_schema_mismatch_message,
)
from tests.spark import spark


# ---------------------------------------------------------------------------
# cell_to_string
# ---------------------------------------------------------------------------
class TestCellToString:
    def test_none_returns_null(self):
        assert cell_to_string(None) == "null"

    def test_string(self):
        assert cell_to_string("hello") == "hello"

    def test_int(self):
        assert cell_to_string(42) == "42"

    def test_bytes(self):
        assert cell_to_string(b"\x00\xff") == "[00 FF]"

    def test_list(self):
        assert cell_to_string([1, 2, 3]) == "[1, 2, 3]"

    def test_tuple(self):
        assert cell_to_string((1, 2)) == "[1, 2]"

    def test_truncation(self):
        long = "a" * 100
        result = cell_to_string(long, truncate=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_row(self):
        r = Row(name="alice", age=30)
        result = cell_to_string(r)
        assert "name -> alice" in result
        assert "age -> 30" in result


# ---------------------------------------------------------------------------
# SeparateLines output format
# ---------------------------------------------------------------------------
class TestSeparateLinesFormat:
    def test_equal_dfs_no_error(self):
        df1 = spark.createDataFrame([Row(a=1, b="x"), Row(a=2, b="y")])
        df2 = spark.createDataFrame([Row(a=1, b="x"), Row(a=2, b="y")])
        formats = FormattingConfig(output_format="separate_lines")
        assert_df_equality(df1, df2, formats=formats)

    def test_unequal_dfs_raises(self):
        df1 = spark.createDataFrame([Row(a=1, b="x")])
        df2 = spark.createDataFrame([Row(a=1, b="z")])
        formats = FormattingConfig(output_format="separate_lines")
        with pytest.raises(DataFramesNotEqualError):
            assert_df_equality(df1, df2, formats=formats)

    def test_different_row_count(self):
        df1 = spark.createDataFrame([Row(a=1), Row(a=2)])
        df2 = spark.createDataFrame([Row(a=1)])
        formats = FormattingConfig(output_format="separate_lines")
        with pytest.raises(DataFramesNotEqualError):
            assert_df_equality(df1, df2, formats=formats)

    def test_with_nan_equality(self):
        df1 = spark.createDataFrame([Row(a=float("nan"))])
        df2 = spark.createDataFrame([Row(a=float("nan"))])
        formats = FormattingConfig(output_format="separate_lines")
        assert_df_equality(df1, df2, allow_nan_equality=True, formats=formats)

    def test_with_null_values(self):
        schema = StructType([StructField("a", IntegerType(), True), StructField("b", StringType(), True)])
        df1 = spark.createDataFrame([(1, None)], schema=schema)
        df2 = spark.createDataFrame([(1, "x")], schema=schema)
        formats = FormattingConfig(output_format="separate_lines")
        with pytest.raises(DataFramesNotEqualError):
            assert_df_equality(df1, df2, formats=formats)

    def test_side_by_side_default(self):
        """Default format should be SIDE_BY_SIDE."""
        df1 = spark.createDataFrame([Row(a=1)])
        df2 = spark.createDataFrame([Row(a=2)])
        with pytest.raises(DataFramesNotEqualError):
            assert_df_equality(df1, df2)


class TestSeparateLinesRaw:
    def test_show_diff_separate_lines_equal(self):
        rows = [Row(a=1, b="x")]
        all_equal, msg = show_diff_separate_lines(["a", "b"], rows, rows)
        assert all_equal

    def test_show_diff_separate_lines_mismatch(self):
        r1 = [Row(a=1, b="x")]
        r2 = [Row(a=1, b="z")]
        all_equal, msg = show_diff_separate_lines(["a", "b"], r1, r2)
        assert not all_equal

    def test_show_diff_side_by_side_equal(self):
        rows = [Row(a=1, b="x")]
        formats = FormattingConfig()
        all_equal, msg = show_diff_side_by_side(rows, rows, formats=formats)
        assert all_equal

    def test_show_diff_side_by_side_mismatch(self):
        r1 = [Row(a=1, b="x")]
        r2 = [Row(a=1, b="z")]
        formats = FormattingConfig()
        all_equal, msg = show_diff_side_by_side(r1, r2, formats=formats)
        assert not all_equal


# ---------------------------------------------------------------------------
# Schema tree comparer
# ---------------------------------------------------------------------------
class TestSchemaTreeComparer:
    def test_equal_schemas(self):
        s = StructType([StructField("a", IntegerType(), True), StructField("b", StringType(), True)])
        tree = build_comparison(s, s)
        assert are_fields_equal(tree)

    def test_different_types(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", StringType(), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_different_names(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("b", IntegerType(), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_different_nullable_ignored(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", IntegerType(), False)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree, ignore_nullable=False)
        assert are_fields_equal(tree, ignore_nullable=True)

    def test_missing_field_right(self):
        s1 = StructType([StructField("a", IntegerType(), True), StructField("b", StringType(), True)])
        s2 = StructType([StructField("a", IntegerType(), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_missing_field_left(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", IntegerType(), True), StructField("b", StringType(), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_nested_struct(self):
        inner1 = StructType([StructField("x", IntegerType(), True)])
        inner2 = StructType([StructField("x", StringType(), True)])
        s1 = StructType([StructField("a", inner1, True)])
        s2 = StructType([StructField("a", inner2, True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_array_type(self):
        s1 = StructType([StructField("a", ArrayType(IntegerType()), True)])
        s2 = StructType([StructField("a", ArrayType(StringType()), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_map_type(self):
        s1 = StructType([StructField("a", MapType(StringType(), IntegerType()), True)])
        s2 = StructType([StructField("a", MapType(StringType(), StringType()), True)])
        tree = build_comparison(s1, s2)
        assert not are_fields_equal(tree)

    def test_match_by_name(self):
        s1 = StructType([StructField("b", StringType(), True), StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", IntegerType(), True), StructField("b", StringType(), True)])
        tree = build_comparison(s1, s2, match_field_by_name=True)
        assert are_fields_equal(tree)

    def test_tree_mismatch_message(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", StringType(), True)])
        tree = build_comparison(s1, s2)
        msg = tree_schema_mismatch_message(tree)
        assert "Actual Schema" in msg
        assert "Expected Schema" in msg


class TestSchemaOutputFormat:
    def test_tree_format_detects_mismatch(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", StringType(), True)])
        formats = FormattingConfig(schema_output_format="tree")
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2, formats=formats)

    def test_tree_format_passes_for_equal(self):
        s = StructType([StructField("a", IntegerType(), True)])
        formats = FormattingConfig(schema_output_format="tree")
        assert_schema_equality(s, s, formats=formats)

    def test_tree_format_ignore_nullable(self):
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", IntegerType(), False)])
        formats = FormattingConfig(schema_output_format="tree")
        assert_schema_equality(s1, s2, ignore_nullable=True, formats=formats)

    def test_table_format_default(self):
        """Default schema output format should be TABLE (existing behavior)."""
        s1 = StructType([StructField("a", IntegerType(), True)])
        s2 = StructType([StructField("a", StringType(), True)])
        with pytest.raises(SchemasNotEqualError):
            assert_schema_equality(s1, s2)


# ---------------------------------------------------------------------------
# FormattingConfig enum properties
# ---------------------------------------------------------------------------
class TestFormattingConfigEnums:
    def test_default_output_format(self):
        fc = FormattingConfig()
        assert fc.output_format == DataFrameDiffOutputFormat.SIDE_BY_SIDE

    def test_default_schema_output_format(self):
        fc = FormattingConfig()
        assert fc.schema_output_format == OutputFormat.TABLE

    def test_set_output_format_string(self):
        fc = FormattingConfig(output_format="separate_lines")
        assert fc.output_format == DataFrameDiffOutputFormat.SEPARATE_LINES

    def test_set_output_format_enum(self):
        fc = FormattingConfig(output_format=DataFrameDiffOutputFormat.SEPARATE_LINES)
        assert fc.output_format == DataFrameDiffOutputFormat.SEPARATE_LINES

    def test_set_schema_output_format_string(self):
        fc = FormattingConfig(schema_output_format="tree")
        assert fc.schema_output_format == OutputFormat.TREE

    def test_set_schema_output_format_enum(self):
        fc = FormattingConfig(schema_output_format=OutputFormat.TREE)
        assert fc.schema_output_format == OutputFormat.TREE


# ---------------------------------------------------------------------------
# Integration: full pipeline with both new output formats
# ---------------------------------------------------------------------------
class TestFullIntegration:
    def test_separate_lines_with_tree_schema(self):
        """Both new formats together."""
        df1 = spark.createDataFrame([Row(a=1, b="x")])
        df2 = spark.createDataFrame([Row(a=1, b="x")])
        formats = FormattingConfig(output_format="separate_lines", schema_output_format="tree")
        assert_df_equality(df1, df2, formats=formats)

    def test_separate_lines_with_ignore_column_order(self):
        df1 = spark.createDataFrame([Row(a=1, b="x")])
        df2 = spark.createDataFrame([Row(a=1, b="x")])
        formats = FormattingConfig(output_format="separate_lines")
        assert_df_equality(df1, df2, ignore_column_order=True, formats=formats)

    def test_separate_lines_with_ignore_row_order(self):
        df1 = spark.createDataFrame([Row(a=1), Row(a=2)])
        df2 = spark.createDataFrame([Row(a=2), Row(a=1)])
        formats = FormattingConfig(output_format="separate_lines")
        assert_df_equality(df1, df2, ignore_row_order=True, formats=formats)

    def test_schema_mismatch_with_tree_format(self):
        df1 = spark.createDataFrame([Row(a=1)])
        df2 = spark.createDataFrame([Row(a="x")])
        formats = FormattingConfig(schema_output_format="tree")
        with pytest.raises(SchemasNotEqualError):
            assert_df_equality(df1, df2, formats=formats)

    def test_chispa_class_with_new_formats(self):
        """Test through the Chispa class."""
        from chispa import Chispa

        formats = FormattingConfig(output_format="separate_lines", schema_output_format="tree")
        c = Chispa(formats=formats)
        df1 = spark.createDataFrame([Row(a=1, b="x")])
        df2 = spark.createDataFrame([Row(a=1, b="x")])
        c.assert_df_equality(df1, df2)
