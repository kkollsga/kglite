#!/usr/bin/env python3
"""Build a knowledge graph from CSV files with pandas.

Demonstrates: pd.read_csv -> add_nodes / add_connections, Cypher queries.

Domain: a tiny org chart — employees, departments, and reporting lines.
Sample CSVs are written to a temporary directory so the example is
self-contained; swap in your own CSV paths to use your data.
"""

from pathlib import Path
import tempfile

import pandas as pd

import kglite

# -- Write sample CSVs to a temp directory ---------------------------------

data_dir = Path(tempfile.mkdtemp(prefix="csv_to_graph_"))

pd.DataFrame(
    {
        "emp_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dan", "Erin"],
        "dept_id": [10, 10, 20, 20, 20],
        "salary": [120000, 95000, 105000, 88000, 99000],
    }
).to_csv(data_dir / "employees.csv", index=False)

pd.DataFrame(
    {
        "dept_id": [10, 20],
        "dept_name": ["Engineering", "Sales"],
    }
).to_csv(data_dir / "departments.csv", index=False)

# reporting lines: employee -> manager (both are employees)
pd.DataFrame(
    {
        "emp_id": [2, 3, 4, 5],
        "manager_id": [1, 1, 3, 3],
    }
).to_csv(data_dir / "reports_to.csv", index=False)

# -- Load CSVs with pandas and build the graph -----------------------------

graph = kglite.KnowledgeGraph()

employees = pd.read_csv(data_dir / "employees.csv")
departments = pd.read_csv(data_dir / "departments.csv")
reports_to = pd.read_csv(data_dir / "reports_to.csv")

graph.add_nodes(employees, "Employee", "emp_id", "name")
graph.add_nodes(departments, "Department", "dept_id", "dept_name")

graph.add_connections(
    employees[["emp_id", "dept_id"]],
    "WORKS_IN",
    "Employee",
    "emp_id",
    "Department",
    "dept_id",
)
graph.add_connections(reports_to, "REPORTS_TO", "Employee", "emp_id", "Employee", "manager_id")

schema = graph.schema()
print(f"Built: {schema['node_count']} nodes, {schema['edge_count']} edges")

# -- Example queries -------------------------------------------------------

print("\n--- Headcount and payroll per department ---")
for row in graph.cypher("""
    MATCH (e:Employee)-[:WORKS_IN]->(d:Department)
    RETURN d.title AS dept, count(e) AS headcount, sum(e.salary) AS payroll
    ORDER BY payroll DESC
"""):
    print(f"  {row['dept']}: {row['headcount']} people, payroll {row['payroll']}")

print("\n--- Direct reports of Alice ---")
for row in graph.cypher("""
    MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee {title: 'Alice'})
    RETURN e.title AS report
    ORDER BY report
"""):
    print(f"  {row['report']}")

print("\n--- Reporting chains (up to 2 hops) ---")
for row in graph.cypher("""
    MATCH (e:Employee)-[:REPORTS_TO*1..2]->(boss:Employee)
    RETURN e.title AS employee, boss.title AS reports_up_to
    ORDER BY employee, reports_up_to
"""):
    print(f"  {row['employee']} -> {row['reports_up_to']}")
