"""Contract checks for every first-class ``extensions.*`` JSON Schema.

The guide promises that the directory is a complete machine-readable index.
Keep this dependency-free so the contract runs in the ordinary Python matrix.
Parser-specific accept/reject behavior remains covered beside each Rust parser.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas" / "extensions"
GUIDE = ROOT / "docs" / "python" / "guides" / "mcp-servers.md"
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


def _schemas() -> dict[str, dict[str, object]]:
    return {path.name: json.loads(path.read_text(encoding="utf-8")) for path in sorted(SCHEMA_DIR.glob("*.json"))}


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    assert reference.startswith("#/"), f"fixture validator only supports local refs: {reference}"
    current: Any = root
    for component in reference[2:].split("/"):
        current = current[component.replace("~1", "/").replace("~0", "~")]
    return current


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    return type(left) is type(right) and left == right


def _has_json_type(instance: Any, expected: str) -> bool:
    return {
        "null": instance is None,
        "boolean": isinstance(instance, bool),
        "object": isinstance(instance, dict),
        "array": isinstance(instance, list),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "integer": (
            isinstance(instance, int)
            and not isinstance(instance, bool)
            or isinstance(instance, float)
            and instance.is_integer()
        ),
        "string": isinstance(instance, str),
    }[expected]


def _schema_accepts(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    """Evaluate the JSON-Schema keywords used by cypher_recipes fixtures.

    This intentionally small evaluator keeps the ordinary test matrix
    dependency-free while making the published schema's acceptance contract
    executable. Rust tests separately cover query parsing and runtime values.
    """

    if "$ref" in schema and not _schema_accepts(instance, _resolve_ref(root, schema["$ref"]), root):
        return False
    if "const" in schema and not _json_equal(instance, schema["const"]):
        return False
    if "enum" in schema and not any(_json_equal(instance, option) for option in schema["enum"]):
        return False
    if "oneOf" in schema:
        if sum(_schema_accepts(instance, option, root) for option in schema["oneOf"]) != 1:
            return False

    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if expected_types is not None and not any(_has_json_type(instance, expected) for expected in expected_types):
        return False

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            return False
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            return False

    if isinstance(instance, dict):
        if len(instance) < schema.get("minProperties", 0):
            return False
        if any(required not in instance for required in schema.get("required", [])):
            return False
        if "propertyNames" in schema and any(
            not _schema_accepts(name, schema["propertyNames"], root) for name in instance
        ):
            return False
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties:
                if not _schema_accepts(value, properties[name], root):
                    return False
                continue
            additional = schema.get("additionalProperties", {})
            if additional is False:
                return False
            if isinstance(additional, dict) and not _schema_accepts(value, additional, root):
                return False

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            return False
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            return False
        if schema.get("uniqueItems") and any(
            _json_equal(value, previous) for index, value in enumerate(instance) for previous in instance[:index]
        ):
            return False
        if "items" in schema and any(not _schema_accepts(value, schema["items"], root) for value in instance):
            return False

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            return False
        if "maximum" in schema and instance > schema["maximum"]:
            return False
    return True


def test_every_extension_schema_has_canonical_identity_and_guide_inventory() -> None:
    schemas = _schemas()
    guide = GUIDE.read_text(encoding="utf-8")

    assert schemas, "extension schema directory must not be empty"
    assert len({schema["$id"] for schema in schemas.values()}) == len(schemas)
    for filename, schema in schemas.items():
        stem = Path(filename).stem
        assert schema["$schema"] == DRAFT_2020_12
        assert schema["$id"] == f"https://kglite.readthedocs.io/schemas/extensions/{filename}"
        assert schema["title"] == f"extensions.{stem}"
        assert filename in guide, f"{filename} is missing from the MCP guide schema inventory"


def test_recipe_schema_declares_the_closed_runtime_subset() -> None:
    schema = _schemas()["cypher_recipes.json"]
    definitions = schema["$defs"]
    parameter = definitions["parameterSchema"]
    root = definitions["rootParameterSchema"]

    expected_keywords = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "additionalProperties",
        "description",
    }
    assert parameter["additionalProperties"] is False
    assert set(parameter["properties"]) == expected_keywords
    assert root["properties"]["type"] == {"const": "object"}
    assert root["properties"]["additionalProperties"] == {"const": False}
    assert root["required"] == ["type", "properties", "required", "additionalProperties"]


def test_recipe_schema_supports_nullable_type_arrays_and_rejects_workflow_fields() -> None:
    schema = _schemas()["cypher_recipes.json"]
    definitions = schema["$defs"]

    type_options = definitions["parameterSchema"]["properties"]["type"]["oneOf"]
    assert type_options[1]["uniqueItems"] is True
    assert definitions["typeName"]["enum"] == [
        "null",
        "boolean",
        "object",
        "array",
        "number",
        "integer",
        "string",
    ]
    assert definitions["recipe"]["additionalProperties"] is False
    assert definitions["query"]["additionalProperties"] is False


def test_recipe_schema_acceptance_matches_structural_parser_fixtures() -> None:
    schema = _schemas()["cypher_recipes.json"]
    valid = {
        "code_review": {
            "description": "Exact code review operations.",
            "queries": {
                "resolve_function": {
                    "description": "Resolve a Function by qualified name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "qualified_name": {
                                "type": ["string", "null"],
                                "description": "Qualified function name, or null when explicitly requested.",
                            }
                        },
                        "required": ["qualified_name"],
                        "additionalProperties": False,
                    },
                    "cypher": "RETURN $qualified_name AS qualified_name",
                }
            },
        }
    }
    assert _schema_accepts({}, schema, schema), "an empty catalog is the documented disabled shape"
    assert _schema_accepts(valid, schema, schema)

    invalid: list[dict[str, Any]] = []
    for field in ("description",):
        fixture = deepcopy(valid)
        fixture["code_review"][field] = " \t\n"
        invalid.append(fixture)
    for field in ("description", "cypher"):
        fixture = deepcopy(valid)
        fixture["code_review"]["queries"]["resolve_function"][field] = " \t\n"
        invalid.append(fixture)

    fixture = deepcopy(valid)
    fixture["bad-name"] = fixture.pop("code_review")
    invalid.append(fixture)
    fixture = deepcopy(valid)
    fixture["code_review"]["workflow"] = ["resolve_function"]
    invalid.append(fixture)
    fixture = deepcopy(valid)
    fixture["code_review"]["queries"] = {}
    invalid.append(fixture)
    fixture = deepcopy(valid)
    fixture["code_review"]["queries"]["resolve_function"]["parameters"]["additionalProperties"] = True
    invalid.append(fixture)
    fixture = deepcopy(valid)
    fixture["code_review"]["queries"]["resolve_function"]["parameters"]["properties"]["qualified_name"]["pattern"] = (
        ".*"
    )
    invalid.append(fixture)

    for fixture in invalid:
        assert not _schema_accepts(fixture, schema, schema), fixture
