"""
Recursive schema tree comparison, ported from spark-fast-tests FieldComparison/FieldInfo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from pyspark.sql.types import ArrayType, DataType, MapType, StructField, StructType

from chispa.formatting import format_string
from chispa.formatting.formats import Color, Format


@dataclass
class FieldInfo:
    type_name: str
    nullable: bool
    metadata: dict
    contains_null: Optional[bool] = None
    name: Optional[str] = None

    @classmethod
    def from_struct_field(cls, sf: StructField) -> FieldInfo:
        return cls(type_name=sf.dataType.typeName(), nullable=sf.nullable, metadata=sf.metadata, name=sf.name)

    @classmethod
    def from_data_type(cls, dt: DataType) -> FieldInfo:
        if isinstance(dt, ArrayType):
            return cls(type_name=dt.typeName(), nullable=True, metadata={}, contains_null=dt.containsNull)
        if isinstance(dt, MapType):
            return cls(type_name=dt.typeName(), nullable=True, metadata={}, contains_null=dt.valueContainsNull)
        return cls(type_name=dt.typeName(), nullable=True, metadata={})

    @classmethod
    def for_array_element(cls, array_type: ArrayType) -> FieldInfo:
        et = array_type.elementType
        if isinstance(et, ArrayType):
            return cls(type_name=et.typeName(), nullable=True, metadata={}, contains_null=et.containsNull)
        if isinstance(et, MapType):
            return cls(type_name=et.typeName(), nullable=True, metadata={}, contains_null=et.valueContainsNull)
        return cls(type_name=et.typeName(), nullable=array_type.containsNull, metadata={})


@dataclass
class FieldComparison:
    name: str
    left_info: Optional[FieldInfo] = None
    right_info: Optional[FieldInfo] = None
    children: Sequence[FieldComparison] = field(default_factory=list)


ELEMENT = "element"
KEY = "key"
VALUE = "value"

_RED_FMT = Format(Color.RED)
_GREEN_FMT = Format(Color.GREEN)
_GRAY_FMT = Format(Color.DARK_GRAY)


def build_comparison(s1: StructType, s2: StructType, match_field_by_name: bool = False) -> list[FieldComparison]:
    if match_field_by_name:
        left_fields = {f.name: f for f in s1.fields}
        right_fields = {f.name: f for f in s2.fields}
    else:
        left_fields = {str(i): f for i, f in enumerate(s1.fields)}
        right_fields = {str(i): f for i, f in enumerate(s2.fields)}

    left_keys = list(left_fields.keys())
    right_only_keys = [k for k in right_fields if k not in left_keys]

    return [
        FieldComparison(
            name,
            FieldInfo.from_struct_field(left_sf) if (left_sf := left_fields.get(name)) else None,
            FieldInfo.from_struct_field(right_sf) if (right_sf := right_fields.get(name)) else None,
            _build_child_comparisons(
                left_sf.dataType if left_sf else None,
                right_sf.dataType if right_sf else None,
                match_field_by_name,
            ),
        )
        for name in left_keys + right_only_keys
    ]


def _build_child_comparisons(
    left_type: DataType | None, right_type: DataType | None, match_field_by_name: bool
) -> list[FieldComparison]:
    if isinstance(left_type, ArrayType) and isinstance(right_type, ArrayType):
        return [FieldComparison(ELEMENT, FieldInfo.for_array_element(left_type), FieldInfo.for_array_element(right_type),
                                _build_child_comparisons(left_type.elementType, right_type.elementType, match_field_by_name))]
    if isinstance(left_type, ArrayType) and right_type is None:
        return [FieldComparison(ELEMENT, FieldInfo.for_array_element(left_type), None,
                                _build_child_comparisons(left_type.elementType, None, match_field_by_name))]
    if left_type is None and isinstance(right_type, ArrayType):
        return [FieldComparison(ELEMENT, None, FieldInfo.for_array_element(right_type),
                                _build_child_comparisons(None, right_type.elementType, match_field_by_name))]

    if isinstance(left_type, StructType) and isinstance(right_type, StructType):
        return build_comparison(left_type, right_type, match_field_by_name)
    if isinstance(left_type, StructType) and right_type is None:
        return build_comparison(left_type, StructType([]), match_field_by_name)
    if left_type is None and isinstance(right_type, StructType):
        return build_comparison(StructType([]), right_type, match_field_by_name)

    if isinstance(left_type, MapType) and isinstance(right_type, MapType):
        return [
            FieldComparison(KEY, FieldInfo.from_data_type(left_type.keyType), FieldInfo.from_data_type(right_type.keyType),
                            _build_child_comparisons(left_type.keyType, right_type.keyType, match_field_by_name)),
            FieldComparison(VALUE, FieldInfo.from_data_type(left_type.valueType), FieldInfo.from_data_type(right_type.valueType),
                            _build_child_comparisons(left_type.valueType, right_type.valueType, match_field_by_name)),
        ]
    if isinstance(left_type, MapType) and right_type is None:
        return [
            FieldComparison(KEY, FieldInfo.from_data_type(left_type.keyType), None, _build_child_comparisons(left_type.keyType, None, match_field_by_name)),
            FieldComparison(VALUE, FieldInfo.from_data_type(left_type.valueType), None, _build_child_comparisons(left_type.valueType, None, match_field_by_name)),
        ]
    if left_type is None and isinstance(right_type, MapType):
        return [
            FieldComparison(KEY, None, FieldInfo.from_data_type(right_type.keyType), _build_child_comparisons(None, right_type.keyType, match_field_by_name)),
            FieldComparison(VALUE, None, FieldInfo.from_data_type(right_type.valueType), _build_child_comparisons(None, right_type.valueType, match_field_by_name)),
        ]
    return []


def are_fields_equal(
    fields: list[FieldComparison],
    ignore_nullable: bool = False,
    ignore_column_names: bool = False,
    ignore_metadata: bool = True,
) -> bool:
    return all(_field_equal(fc, ignore_nullable, ignore_column_names, ignore_metadata) for fc in fields)


def _field_equal(
    fc: FieldComparison, ignore_nullable: bool, ignore_column_names: bool, ignore_metadata: bool,
) -> bool:
    left, right = fc.left_info, fc.right_info
    if left is not None and right is not None:
        if left.type_name != right.type_name:
            return False
        if not ignore_column_names and left.name != right.name:
            return False
        if not ignore_nullable and left.nullable != right.nullable:
            return False
        if not ignore_metadata and left.metadata != right.metadata:
            return False
    elif not (left is None and right is None):
        return False
    return all(_field_equal(child, ignore_nullable, ignore_column_names, ignore_metadata) for child in fc.children)


TREE_GAP = 6


def tree_schema_mismatch_message(
    field_comparisons: list[FieldComparison],
    ignore_nullable: bool = False,
    ignore_metadata: bool = True,
) -> str:
    def _depth_prefix(indent: int) -> str:
        return "".join("|    " for _ in range(indent)) + "|--"

    def _colored(text: str, fmt: Format) -> str:
        return format_string(text, fmt)

    def _colored_field_info(
        info: FieldInfo | None, other: FieldInfo | None, fallback_name: str, indent: int, is_left: bool
    ) -> str:
        if info is None:
            return ""
        prefix = _depth_prefix(indent)
        fname = info.name if info.name else fallback_name
        tname = info.type_name
        nullable_str = "" if ignore_nullable else f"(nullable = {info.nullable})"
        contains_null_str = f"(containsNull = {info.contains_null})" if info.contains_null is not None else ""
        description = contains_null_str if contains_null_str else nullable_str

        color_fmt = _RED_FMT if is_left else _GREEN_FMT

        if other is None:
            parts = [_colored(prefix, color_fmt), " ", _colored(fname, color_fmt), " : ", _colored(tname, color_fmt)]
            if description:
                parts += [" ", _colored(description, color_fmt)]
            return "".join(parts)

        prefix_s = _colored(prefix, _GRAY_FMT)
        name_c = _GRAY_FMT if info.name == other.name else color_fmt
        type_c = _GRAY_FMT if info.type_name == other.type_name else color_fmt
        parts = [prefix_s, " ", _colored(fname, name_c), " : ", _colored(tname, type_c)]
        if description:
            desc_matches = (info.nullable == other.nullable or ignore_nullable) and info.contains_null == other.contains_null
            desc_c = _GRAY_FMT if desc_matches else color_fmt
            parts += [" ", _colored(description, desc_c)]
        return "".join(parts)

    def _strip_ansi(s: str) -> str:
        return re.sub(r"\033\[[0-9;]*m", "", s)

    def _calc_max_width(comparisons: list[FieldComparison], indent: int) -> int:
        if not comparisons:
            return 0
        widths = [
            max(
                len(_strip_ansi(_colored_field_info(fc.left_info, fc.right_info, fc.name, indent, True))),
                _calc_max_width(list(fc.children), indent + 1) if fc.children else 0,
            )
            for fc in comparisons
        ]
        return max(widths)

    def _format_line(fc: FieldComparison, indent: int, max_width: int) -> str:
        left_str = _colored_field_info(fc.left_info, fc.right_info, fc.name, indent, True)
        right_str = _colored_field_info(fc.right_info, fc.left_info, fc.name, indent, False)
        gap = max_width + TREE_GAP
        if not left_str:
            return " " * gap + right_str
        if not right_str:
            return left_str
        padding = gap - len(_strip_ansi(left_str))
        return left_str + " " * padding + right_str

    def _build_lines(comparisons: list[FieldComparison], indent: int, max_width: int) -> list[str]:
        return [
            line
            for fc in comparisons
            for line in [_format_line(fc, indent, max_width)]
            + (_build_lines(list(fc.children), indent + 1, max_width) if fc.children else [])
        ]

    max_width = _calc_max_width(field_comparisons, 0)
    gap = max_width + TREE_GAP
    header = "Actual Schema".ljust(gap) + "Expected Schema"
    body = "\n".join(_build_lines(field_comparisons, 0, max_width))
    return f"\n{header}\n{body}"
