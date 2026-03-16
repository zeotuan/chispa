from __future__ import annotations

from enum import Enum


class OutputFormat(str, Enum):
    TABLE = "table"
    TREE = "tree"


class DataFrameDiffOutputFormat(str, Enum):
    SIDE_BY_SIDE = "side_by_side"
    SEPARATE_LINES = "separate_lines"


class TypeName(str, Enum):
    ARRAY = "array"
    STRUCT = "struct"
